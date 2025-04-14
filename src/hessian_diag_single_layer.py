import argparse
import time

import torch.nn.functional as F

from utils import *
from data import get_cached_wikitext2


def compute_hessian_diag_hutchinson(model_name, layer_name, block_index, model_input_bs, seqlen, b, vhp_samples, seed, cache_dir):
    set_seed(seed)

    disable_non_differential_modules()

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    _, testloader = get_cached_wikitext2(tokenizer=tokenizer, seqlen=seqlen, seed=seed)

    block = get_all_blocks(model)[block_index]
    layer = get_nested_attr(block, layer_name)

    params = layer.weight.clone().requires_grad_(True)

    samples_in_dataset = testloader.input_ids.numel() // seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

    model_input_bs = min(b, model_input_bs)
    assert b % model_input_bs == 0, "`b` should be divisible by `model_input_bs`"
    num_batches = b // model_input_bs

    print("Number of batches =", num_batches)

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

    def get_partial_ppl_fn(i_start):
        def partial_ppl_fn(x):
            # Define custom forward method for q_proj
            def custom_forward(self, inpt):
                return F.linear(input=inpt, weight=x, bias=self.bias)

            # Monkey-patch q_proj's forward method
            layer.forward = custom_forward.__get__(layer, type(layer))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs, debug=False, seqlen=seqlen)

        return partial_ppl_fn

    # NOTICE: here we rely on the additive property of the PPL function (Corollary 7.2 in a technical report)
    hess_diag = torch.zeros_like(params, device=params.device)
    draw_progress_bar(0, num_batches)
    for k in range(num_batches):
        ppl_fn = get_partial_ppl_fn(i_start=k * model_input_bs)

        diag_estimate = torch.zeros_like(params)

        for j in range(vhp_samples):
            v = hadamard_vector(size=params.numel(), block_size=32).reshape(diag_estimate.shape)
            v = v.to(device=params.device)

            _, Hv = torch.autograd.functional.vhp(ppl_fn, (params,), (v,))
            diag_estimate += Hv[0] * v / num_batches  # Diagonal approximation

            draw_progress_bar(k*vhp_samples + j + 1, num_batches*vhp_samples)

        hess_diag = hess_diag.to(device=diag_estimate.device)
        hess_diag += diag_estimate / vhp_samples


    # NOTICE: This code is written for one single experiment in order to understand how errors evolve over time
    '''
    print("Started")
    start_time = time.perf_counter()
    
    real_diag_first_row = torch.diag(torch.load("data/hessian_q_proj_b_60_t_768.pt"))
    diag_estimate = torch.zeros_like(params)

    diffs = [torch.tensor([1.0])]
    errors = [torch.tensor([1.0])]
    times = [0]
    for i in range(vhp_samples):
        v = hadamard_vector(size=params.numel(), block_size=32).reshape(params.shape)
        v = v.to(device=params.device)

        prev_diag_estimate = diag_estimate.clone()

        for k in range(num_batches):
            ppl_fn = get_partial_ppl_fn(i_start=k * model_input_bs)
            _, Hv = torch.autograd.functional.vhp(ppl_fn, (params,), (v,))
            diag_estimate += Hv[0] * v / num_batches

        errors.append(torch.linalg.norm(diag_estimate[0]/(i+1) - real_diag_first_row) / torch.linalg.norm(real_diag_first_row))
        times.append(time.perf_counter() - start_time)

        if i != 0:
            diffs.append(
                torch.linalg.norm(prev_diag_estimate / i - diag_estimate / (i + 1)) / torch.linalg.norm(diag_estimate / (i + 1))
            )

        print("-----------------------------")
        print("vhp_samples =", i+1)
        print("diff =", diffs[-1].item())
        print("errors =", errors[-1].item())
        print("time =", times[-1])

        torch.save(torch.tensor(diffs), "data/diag_hessian/relative_l2_diffs.pt")
        torch.save(torch.tensor(errors), "data/diag_hessian/partial_relative_l2_loss.pt")
        torch.save(torch.tensor(times), "data/diag_hessian/times.pt")
    '''

    return hess_diag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layer_name", type=str, default="self_attn.q_proj")
    parser.add_argument("--vhp_samples", type=int, default=50)
    parser.add_argument("--block_index", type=int, default=0)
    parser.add_argument("--b", type=int, default=10)
    parser.add_argument("--model_input_bs", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()

    start_t = time.perf_counter()
    hess_diag = compute_hessian_diag_hutchinson(model_name=args.model, layer_name=args.layer_name,
                                                vhp_samples=args.vhp_samples, block_index=args.block_index,
                                                b=args.b, model_input_bs=args.model_input_bs, seqlen=args.seqlen,
                                                seed=args.seed, cache_dir=args.cache_dir)
    print("Computation time =", time.perf_counter() - start_t)

    out_path_prefix = "data/" + args.model + "/diag_hessian/hessian_diag_" + args.layer_name + "_block" + str(args.block_index) + "_vhp_samples" + str(args.vhp_samples) + "_b" + str(args.b) + "_seed" + str(args.seed)
    plot_heatmap(torch.abs(hess_diag), out_path_prefix + '.pdf')
    torch.save(hess_diag, out_path_prefix + ".pt")
