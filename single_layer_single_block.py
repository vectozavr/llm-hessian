import argparse
import time

import torch.nn.functional as F

from torch.autograd.functional import hessian
from utils import *
from data import get_cached_wikitext2


def compute_hessian_single_layer_single_block(model_name, layer_name, block_index, t, b, model_input_bs, seqlen, seed, cache_dir):
    # Setting seeds for reproducibility
    set_seed(seed)

    disable_non_differential_modules()

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    _, testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    # Access the first transformer layer and q_proj weight matrix
    block = get_all_blocks(model)[block_index]
    layer = get_nested_attr(block, layer_name)
    weight_matrix = layer.weight

    params = weight_matrix[0][:t].clone().requires_grad_(True)
    residuals = weight_matrix[0][t:].clone()
    partial_weights = weight_matrix[1:]

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

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
                concatenated_line = torch.cat((x, residuals), dim=0)
                line_reshaped = concatenated_line.unsqueeze(0)
                full_weights = torch.cat((line_reshaped, partial_weights), dim=0)

                return F.linear(input=inpt, weight=full_weights, bias=self.bias)

            # Monkey-patch forward method
            layer.forward = custom_forward.__get__(layer, type(layer))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs, debug=False, seqlen=seqlen)

        return partial_ppl_fn

    hess = torch.zeros((t, t))

    draw_progress_bar(0, num_batches)
    # NOTICE: here we rely on the additive property of the PPL function (Corollary in a technical report)
    for k in range(num_batches):

        dH = hessian(get_partial_ppl_fn(i_start=k * model_input_bs), params)
        hess = hess.to(device=dH.device)
        hess += dH / num_batches

        draw_progress_bar(k+1, num_batches)

        #torch.save(hess * num_batches / (k+1), "data/diff_bs/b_" + str(k+1) + ".pt")

    return hess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--layer_name", type=str, default="self_attn.q_proj")
    parser.add_argument("--block_index", type=int, default=0)
    parser.add_argument("--t", type=int, default=25)
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--model_input_bs", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()
    print('Computing Hessian for ' + args.layer_name + ' from block ' + str(args.block_index) + ' of ' + args.model)

    start_t = time.perf_counter()
    hess = compute_hessian_single_layer_single_block(model_name=args.model, layer_name=args.layer_name,
                                                     block_index=args.block_index, t=args.t, b=args.b,
                                                     model_input_bs=args.model_input_bs, seqlen=args.seqlen,
                                                     seed=args.seed, cache_dir=args.cache_dir)
    print("Computation time =", time.perf_counter() - start_t)

    # Computation time = 29338.510749154957 sec

    out_path_prefix = "data/hessian_" + args.layer_name + "_block" + str(args.block_index) + "_t" + str(args.t) + "_b" + str(args.b) + "_seed" + str(args.seed)
    plot_heatmap(torch.abs(hess), out_path_prefix + '.pdf')
    torch.save(hess, out_path_prefix + ".pt")
