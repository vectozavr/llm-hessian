import torch.nn.functional as F
import argparse
import time
import math
import torch

from utils import set_seed, get_llm, check_gpus, ppl_function, plot_heatmap
from data import get_cached_wikitext2


def compute_hessian(loss_fn, param_block, d):
    print("Start computing Hessian by finite differences")

    ppl_0 = loss_fn(param_block)
    ppl_d = torch.zeros_like(param_block, device=ppl_0.device)

    d = torch.tensor([d], dtype=torch.double, device=param_block.device)

    n = param_block.numel()
    e = torch.eye(n, device=param_block.device)

    hessian_res = torch.zeros((n, n), device=param_block.device)

    print("Preparing", n, "estimations of perplexity")
    for i in range(n):
        d_i = e[i] * d
        ppl_d[i] = loss_fn(param_block + d_i)

    print("Do main computations (most complex part with", math.ceil((n+1)*n/2), "estimations of perplexity)")

    for i in range(n):
        d_i = e[i] * d
        for j in range(i, n):

            d_j = e[j] * d

            print("ij=(" + str(i) + "," + str(j) + ")")

            _res = (loss_fn(param_block + d_i + d_j).to(device=ppl_0.device) - ppl_d[i] - ppl_d[j] + ppl_0) / d.to(
                device=ppl_0.device) ** 2

            hessian_res[i, j] = _res
            hessian_res[j, i] = _res

    print("Finish computing the Hessian")

    return hessian_res


def compute_hessian_by_central_diff(loss_fn, param_block, d):
    print("Start computing Hessian by central finite differences")

    d = torch.tensor([d], dtype=torch.double, device=param_block.device)

    n = param_block.numel()
    e = torch.eye(n, device=param_block.device)

    hessian_res = torch.zeros((n, n), device=param_block.device)

    print("Do main computations (most complex part with", math.ceil((n+1)*n/2), "estimations of perplexity)")

    for i in range(n):
        d_i = e[i] * d
        for j in range(i, n):
            d_j = e[j] * d

            print("ij=(" + str(i) + "," + str(j) + ")")
            p1 = loss_fn(param_block + d_i + d_j)
            p2 = loss_fn(param_block + d_i - d_j)
            p3 = loss_fn(param_block - d_i + d_j)
            p4 = loss_fn(param_block - d_i - d_j)

            _res = (p1 - p2 - p3 + p4) / (2 * d.to(device=p1.device)) ** 2

            hessian_res[i, j] = _res
            hessian_res[j, i] = _res

    print("Finish computing the Hessian")

    return hessian_res


def compute_hessian_by_finite_diff(model_name, cache_dir, seed, t=25):

    # Setting seeds for reproducibility
    set_seed(seed)

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    # Access the first transformer layer and q_proj weight matrix
    first_layer = model.model.decoder.layers[10]
    self_attn = first_layer.self_attn
    weight_matrix = self_attn.q_proj.weight

    params = weight_matrix[0][:t].clone()
    residual_block = weight_matrix[0][t:].clone()
    partial_weights = weight_matrix[1:]

    batch_size = 140

    def ppl_fn(x):
        # Define custom forward method for q_proj
        def custom_q_proj_forward(self, inpt):
            concatenated_line = torch.cat((x, residual_block), dim=0)
            line_reshaped = concatenated_line.unsqueeze(0)
            full_weights = torch.cat((line_reshaped, partial_weights), dim=0)

            return F.linear(inpt, full_weights, self.bias)

        # Monkey-patch q_proj's forward method
        self_attn.q_proj.forward = custom_q_proj_forward.__get__(self_attn.q_proj, type(self_attn.q_proj))

        return ppl_function(model, testloader, 1, device, nsamples=batch_size)

    hess = compute_hessian(ppl_fn, params, 1e-4)
    #hess = compute_hessian_by_central_diff(loss_fn, param_block, 1e-4)

    return hess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    # Enable double precision
    torch.set_default_dtype(torch.float64)

    print("IMPORTANT: This method is not efficient and not robust. Consider using torch.autograd.functional.hessian "
          "instead.")
    check_gpus()

    start_t = time.perf_counter()
    hess = compute_hessian_by_finite_diff(model_name=args.model, cache_dir=args.cache_dir, seed=args.seed)
    print("Computation time =", time.perf_counter() - start_t)

    plot_heatmap(hess)

