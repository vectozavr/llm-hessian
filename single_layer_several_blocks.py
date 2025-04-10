import torch.nn.functional as F
import argparse
import time

from torch.autograd.functional import hessian
from utils import *
from data import get_cached_wikitext2


def compute_hessian_single_layer_several_blocks(model_name, layer_name, num_blocks, t, b, model_input_bs, seqlen, seed, cache_dir):
    # Setting seeds for reproducibility
    set_seed(seed)

    disable_non_differential_modules()

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    _, testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    params = torch.zeros((t * num_blocks), device=device)
    partial_weights = []
    residuals_blocks = []

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

    model_input_bs = min(b, model_input_bs)
    assert b % model_input_bs == 0, "`b` should be divisible by `model_input_bs`"
    num_batches = b // model_input_bs

    print("Number of batches =", num_batches)

    blocks = get_all_blocks(model)
    for j in range(num_blocks):
        block = blocks[j]
        layer = get_nested_attr(block, layer_name)
        weight_matrix = layer.weight

        # take 'line_size' parameters from the first row of the 'weight_matrix'
        params_in_block = weight_matrix[0][:t].clone().requires_grad_(True)
        residuals = weight_matrix[0][t:].clone()
        partial_weight = weight_matrix[1:]

        params[j*t:(j+1)*t] = params_in_block

        partial_weights.append(partial_weight)
        residuals_blocks.append(residuals)

    def get_partial_ppl_fn(i_start):
        def partial_ppl_fn(x):
            for j in range(num_blocks):
                # Define custom forward method for q_proj

                def custom_forward_j(i_from):
                    def custom_forward(self, inpt):
                        _param_block = x[i_from*t:(i_from+1)*t]
                        _residual_block = residuals_blocks[i_from]
                        concatenated_line = torch.cat((_param_block.to(device=_residual_block.device), _residual_block), dim=0)
                        line_reshaped = concatenated_line.unsqueeze(0)
                        _partial_weight = partial_weights[i_from]
                        full_weights = torch.cat((line_reshaped, _partial_weight), dim=0)

                        return F.linear(inpt, full_weights, self.bias)

                    return custom_forward

                # Monkey-patch forward method
                block = blocks[j]
                layer = get_nested_attr(block, layer_name)
                layer.forward = custom_forward_j(j).__get__(layer, type(layer))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs, debug=False, seqlen=seqlen)

        return partial_ppl_fn

    hess = torch.zeros((t*num_blocks, t*num_blocks))

    draw_progress_bar(0, num_batches)
    # NOTICE: here we rely on the additive property of the PPL function (Corollary in a technical report)
    for k in range(num_batches):
        dH = hessian(get_partial_ppl_fn(i_start=k * model_input_bs), params)
        hess = hess.to(device=dH.device)
        hess += dH / num_batches

        draw_progress_bar(k+1, num_batches)

    return hess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--layer_name", type=str, default="self_attn.q_proj")
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--b", type=int, default=30)
    parser.add_argument("--model_input_bs", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    check_gpus()

    print('Computing Hessian for ' + args.layer_name + ' from blocks (1-' + str(args.num_blocks) + ') of ' + args.model)
    start_t = time.perf_counter()
    hess = compute_hessian_single_layer_several_blocks(model_name=args.model, layer_name=args.layer_name,
                                                       num_blocks=args.num_blocks, t=args.t, b=args.b, seed=args.seed,
                                                       model_input_bs=args.model_input_bs, seqlen=args.seqlen,
                                                       cache_dir=args.cache_dir)
    print("Computation time =", time.perf_counter() - start_t)

    # Computation time = 71551.41185522405 sec

    out_path_prefix = "data/hessian_" + args.layer_name + "_" + str(args.num_blocks) + "_blocks_t" + str(args.t) + "_b" + str(args.b) + "_seed" + str(args.seed)
    plot_heatmap(torch.abs(hess), out_path_prefix + '.pdf')
    torch.save(hess, out_path_prefix + ".pt")


