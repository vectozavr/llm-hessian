import argparse
import time

import torch
import torch.nn.functional as F

from utils import set_seed, get_llm, ppl_function, check_gpus, plot_heatmap, plot_hist
from data import get_cached_wikitext2


def compute_hessian_diag_hutchinson(model_name, cache_dir, seed, block_number=0, model_input_bs=4, b=32, vhp_samples=100):
    set_seed(seed)

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    # Access the first transformer layer and q_proj weight matrix
    layer = model.model.decoder.layers[block_number]
    self_attn = layer.self_attn

    # Full q_proj matrix:
    params = self_attn.q_proj.weight.clone().requires_grad_(True)

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

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
            def custom_q_proj_forward(self, inpt):
                return F.linear(input=inpt, weight=x, bias=self.bias)

            # Monkey-patch q_proj's forward method
            self_attn.q_proj.forward = custom_q_proj_forward.__get__(self_attn.q_proj, type(self_attn.q_proj))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs,
                                debug=False)

        return partial_ppl_fn

    hess_diag = torch.zeros_like(params, device=params.device)

    # NOTICE: here we rely on the additive property of the PPL function (Corollary in a technical report)
    for k in range(num_batches):
        #t1 = time.perf_counter()
        ppl_fn = get_partial_ppl_fn(i_start=k * model_input_bs)

        diag_estimate = torch.zeros_like(params)

        for _ in range(vhp_samples):
            v = hadamard_vector(size=params.numel(), block_size=32).reshape(diag_estimate.shape)
            v = v.to(device=params.device)

            _, Hv = torch.autograd.functional.vhp(ppl_fn, (params,), (v,))
            diag_estimate += Hv[0] * v  # Diagonal approximation

        hess_diag = hess_diag.to(device=diag_estimate.device)

        hess_diag += diag_estimate / (vhp_samples * num_batches)

        #print("dt =", time.perf_counter() - t1)
        print("Processed " + str((k+1)*model_input_bs) + " samples...")

    return hess_diag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()

    vhp_samples_list = [10, 100, 1000, 3000, 5000, 10000, 30000, 50000, 100000]

    for vhp_samples in vhp_samples_list:
        print("vhp_samples =", vhp_samples)

        start_t = time.perf_counter()
        hess_diag = compute_hessian_diag_hutchinson(model_name=args.model,
                                                    cache_dir=args.cache_dir,
                                                    seed=args.seed,
                                                    vhp_samples=vhp_samples)

        torch.save(hess_diag, "data/diag_hessian/hessian_diag_q_proj_vhp_samples_" + str(vhp_samples) + ".pt")

        print("Computation time =", time.perf_counter() - start_t)

    #plot_heatmap(torch.abs(hess_diag)[0][:25].reshape(1, -1))
