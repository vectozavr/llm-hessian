import argparse
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import set_seed, get_llm, ppl_function, check_gpus, plot_heatmap, plot_hist, draw_progress_bar
from data import get_cached_wikitext2


def get_all_blocks(model):
    if "opt" not in model.name_or_path:
        return model.model.layers
    else:
        return model.model.decoder.layers


def prepare_calibration_input(model, dataloader, device, b):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    blocks = get_all_blocks(model)

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
    elif len(model.hf_device_map) == 1:
        device = model.hf_device_map['']

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((b, min(2048, model.seqlen), model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False

    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.reshape((-1, inp.shape[-1])).to(torch.device("cpu"))
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_embeddings' in kwargs:
                cache['position_embeddings'] = kwargs['position_embeddings']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    blocks[0] = Catcher(blocks[0])

    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    blocks[0] = blocks[0].module

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    model.config.use_cache = use_cache

    return inps, attention_mask, position_ids, position_embeddings


def compute_hessian_diag_hutchinson(model_name, cache_dir, seed, block_number=0, b=60, model_input_bs=10, vhp_samples=100):
    set_seed(seed)

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    trainloader, testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed, b=b)

    # Access the first transformer layer and q_proj weight matrix
    block = get_all_blocks(model)[block_number]
    self_attn = block.self_attn

    # Full q_proj matrix:
    params = self_attn.q_proj.weight.clone().requires_grad_(True)

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

    assert b % model_input_bs == 0, "`b` should be divisible by `model_input_bs`"
    num_batches = b // model_input_bs

    print("Number of batches =", num_batches)

    # Prepare input data for the 1 block
    with torch.no_grad():
        inps, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, trainloader, device, b=b)

    block_args = {}
    if attention_mask is not None:
        block_args["attention_mask"] = attention_mask.expand(b, -1, -1, -1)
    if position_ids is not None:
        block_args["position_ids"] = position_ids
    if position_embeddings is not None:
        block_args["position_embeddings"] = position_embeddings

    # compute the real output of the first block
    with torch.no_grad():
        outs = block(inps, **block_args)[0]

    if attention_mask is not None:
        block_args["attention_mask"] = attention_mask.expand(model_input_bs, -1, -1, -1)

    def balanced_block_rademacher(size, block_size=8, device="cuda:0"):
        assert size % block_size == 0, "Size must be a multiple of block_size"

        # Create blocks with equal number of +1 and -1
        blocks = torch.cat([
            torch.ones((size // block_size, block_size // 2), device=device),
            -torch.ones((size // block_size, block_size // 2), device=device)
        ], dim=1)

        # Shuffle within each block
        idx = torch.argsort(torch.rand(blocks.shape, device=device), dim=1)
        blocks = torch.gather(blocks, 1, idx)

        # Reshape to original size
        return blocks.view(-1)

    def hadamard_vector(size, block_size=8):
        _v = balanced_block_rademacher(size, block_size)
        return torch.fft.ifft(torch.fft.fft(_v)).real  # Fast Hadamard-like transform

    def get_partial_loss_fn(i_start):
        def partial_loss_fn(x):
            # Define custom forward method for q_proj
            def custom_q_proj_forward(self, inpt):
                return F.linear(input=inpt, weight=x, bias=self.bias)

            # Monkey-patch q_proj's forward method
            block.self_attn.q_proj.forward = custom_q_proj_forward.__get__(self_attn.q_proj, type(self_attn.q_proj))

            _block_out = block(inps[i_start:i_start+model_input_bs], **block_args)[0]
            _real_out = outs[i_start:i_start+model_input_bs]

            # For some reason this one gives nans in torch.autograd.functional.vhp
            #_loss_val = torch.sum(torch.norm(_block_out - _real_out, dim=(1, 2))) / b
            # So, instead we use this one:
            _loss_val = torch.sum((_block_out - _real_out)**2) / b

            return _loss_val

        return partial_loss_fn

    hess_diag = torch.zeros_like(params, device=params.device)

    for k in range(num_batches):
        # t1 = time.perf_counter()
        partial_loss = get_partial_loss_fn(i_start=k * model_input_bs)

        diag_estimate = torch.zeros_like(params)

        for i in range(vhp_samples):
            v = hadamard_vector(size=params.numel(), block_size=32).reshape(diag_estimate.shape)
            v = v.to(device=params.device)

            _, Hv = torch.autograd.functional.vhp(partial_loss, (params,), (v,))
            diag_estimate += Hv[0] * v  # Diagonal approximation

            draw_progress_bar(i + 1, vhp_samples)

        hess_diag = hess_diag.to(device=diag_estimate.device)
        hess_diag += diag_estimate / vhp_samples

        # print("dt =", time.perf_counter() - t1)
        print("Processed " + str((k + 1) * model_input_bs) + " samples...")

    return hess_diag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()

    start_time = time.perf_counter()
    hess_diag_res = compute_hessian_diag_hutchinson(model_name=args.model,
                                                    cache_dir=args.cache_dir,
                                                    seed=args.seed,
                                                    b=60,
                                                    vhp_samples=5000)
    print("dt =", time.perf_counter() - start_time)

    torch.save(hess_diag_res, "data/diag_hessian/hessian_diag_q_proj_vhp_samples_5000_block_loss.pt")
