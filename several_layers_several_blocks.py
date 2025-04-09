import torch.nn.functional as F
import argparse
import time

from utils import *
from torch.autograd.functional import hessian
from data import get_cached_wikitext2


def compute_hessian_several_layers_several_blocks(model_name, num_layers, num_blocks, t, b, model_input_bs, seqlen, seed, cache_dir):
    # Setting seeds for reproducibility
    set_seed(seed)

    disable_non_differential_modules()

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    _, testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    params = torch.zeros((t * num_layers * num_blocks), device=device, dtype=torch.float32)
    partial_weights_all = []
    residual_blocks_all = []

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

    model_input_bs = min(b, model_input_bs)
    assert b % model_input_bs == 0, "`b` should be divisible by `model_input_bs`"
    num_batches = b // model_input_bs

    print("Number of batches =", num_batches)

    blocks = get_all_blocks(model)
    for i in range(num_blocks):
        # Access the first transformer layer and q_proj weight matrix
        block = blocks[i]
        layers = find_layers(block)
        weights = []
        for j_layer, layer_name in enumerate(layers):
            if j_layer >= num_layers:
                break
            weights.append(layers[layer_name].weight)

        partial_weight_layer = []
        residual_blocks_layer = []
        for j in range(num_layers):
            # for each weight_matrix[i] take 'line_size' parameters from the first row of the 'weight_matrix'
            w = weights[j]
            param_block = w[0][:t].clone().requires_grad_(True)
            partial_weight = w[1:]
            residual_block = w[0][t:].clone()
            partial_weight_layer.append(partial_weight)
            residual_blocks_layer.append(residual_block)

            _from = t * num_layers * i + j * t
            _to = _from + t

            params[_from:_to] = param_block

        partial_weights_all.append(partial_weight_layer)
        residual_blocks_all.append(residual_blocks_layer)

    def get_partial_ppl_fn(i_start):
        def partial_ppl_fn(x):
            for j in range(num_blocks):
                def custom_forward(layer_number, i_from):
                    def _custom_forward(self, inpt):
                        __from = t * num_layers * i_from + layer_number * t
                        __to = __from + t

                        _param_block = x[__from:__to]
                        _residual_block = residual_blocks_all[i_from][layer_number]
                        concatenated_line = torch.cat((_param_block.to(device=_residual_block.device), _residual_block), dim=0)
                        line_reshaped = concatenated_line.unsqueeze(0)
                        _partial_weight = partial_weights_all[i_from][layer_number]
                        full_weights = torch.cat((line_reshaped, _partial_weight), dim=0)

                        return F.linear(inpt, full_weights, self.bias)

                    return _custom_forward

                block = blocks[j]
                layers = find_layers(block)

                for j_layer, layer_name in enumerate(layers):
                    if j_layer >= num_layers:
                        break
                    layer = layers[layer_name]
                    layer.forward = custom_forward(j_layer, j).__get__(layer, type(layer))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs, debug=False)

        return partial_ppl_fn

    hess = torch.zeros((t * num_layers * num_blocks, t * num_layers * num_blocks))

    draw_progress_bar(0, num_batches)
    # NOTICE: here we rely on the additive property of the PPL function (Corollary 7.2 in a technical report)
    for k in range(num_batches):
        dH = hessian(get_partial_ppl_fn(i_start=k * model_input_bs), params)
        hess = hess.to(device=dH.device)
        hess += dH / num_batches

        draw_progress_bar(k+1, num_batches)

    return hess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--b", type=int, default=30)
    parser.add_argument("--model_input_bs", type=int, default=2)
    parser.add_argument("--seqlen", type=str, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    check_gpus()

    start_t = time.perf_counter()
    hess = compute_hessian_several_layers_several_blocks(model_name=args.model, num_layers=args.num_layers,
                                                         num_blocks=args.num_blocks, t=args.t, b=args.b,
                                                         model_input_bs=args.model_input_bs, seqlen=args.seqlen,
                                                         seed=args.seed, cache_dir=args.cache_dir)
    print("Computation time =", time.perf_counter() - start_t)

    # Computation time = 90927.86469955707 (all layers, all blocks, t=25)
    # Computation time = 71593.03141659603 (all layers, 1 block, t=300)

    out_path_prefix = "data/hessian_" + str(args.num_layers) + "_layers_" + str(args.num_blocks) + "_blocks_t" + str(args.t) + "_b" + str(args.b) + "_seed" + str(args.seed)
    plot_heatmap(torch.abs(hess), out_path_prefix + '.pdf')
    torch.save(hess, out_path_prefix + ".pt")