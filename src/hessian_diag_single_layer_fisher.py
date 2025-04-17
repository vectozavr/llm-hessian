import argparse
import time

import torch.nn.functional as F

from utils import *
from data import get_cached_wikitext2


def compute_grad(loss_fn, params):
    loss = loss_fn(params)
    grad, = torch.autograd.grad(loss, params, create_graph=False)
    return grad.detach() ** 2


def compute_hessian_diag_fisher(model_name, layer_name, block_index, model_input_bs, seqlen, b, seed, cache_dir):
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

    def get_partial_ppl_fn(i_start):
        def partial_ppl_fn(x):
            # Define custom forward method for q_proj
            def custom_forward(self, inpt):
                return F.linear(input=inpt, weight=x, bias=self.bias)

            # Monkey-patch q_proj's forward method
            layer.forward = custom_forward.__get__(layer, type(layer))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs,
                                debug=False, seqlen=seqlen)

        return partial_ppl_fn

    # NOTICE: here we rely on the additive property of the PPL function (Corollary 7.2 in a technical report)
    hess_diag = torch.zeros_like(params, device=params.device)
    draw_progress_bar(0, num_batches)
    for k in range(num_batches):
        ppl_fn = get_partial_ppl_fn(i_start=k * model_input_bs)

        diag_estimate = compute_grad(ppl_fn, params)
        hess_diag += diag_estimate / num_batches

    return hess_diag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--layer_name", type=str, default="self_attn.q_proj")
    parser.add_argument("--block_index", type=int, default=0)
    parser.add_argument("--b", type=int, default=60)
    parser.add_argument("--model_input_bs", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()

    start_t = time.perf_counter()
    hess_diag = compute_hessian_diag_fisher(model_name=args.model, layer_name=args.layer_name,
                                            block_index=args.block_index, b=args.b,
                                            model_input_bs=args.model_input_bs, seqlen=args.seqlen,
                                            seed=args.seed, cache_dir=args.cache_dir)
    print("Computation time =", time.perf_counter() - start_t)

    out_path_prefix = "data/" + args.model + "/diag_hessian/hessian_diag_" + args.layer_name + "_block" + str(
        args.block_index) + "_fisher" + "_b" + str(args.b) + "_seed" + str(args.seed)
    plot_heatmap(torch.abs(hess_diag), out_path_prefix + '.pdf')
    torch.save(hess_diag, out_path_prefix + ".pt")
