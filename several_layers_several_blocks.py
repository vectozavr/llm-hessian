import torch.nn.functional as F
import argparse
import torch
import time

from utils import set_seed, get_llm, ppl_function, check_gpus, plot_heatmap
from torch.autograd.functional import hessian
from data import get_cached_wikitext2


def compute_hessian_several_layers_several_blocks(model_name, cache_dir, seed, num_blocks=12, num_layers=6, t=25, model_input_bs=4, b=140):
    # Setting seeds for reproducibility
    set_seed(seed)

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    params = torch.zeros((t * num_layers * num_blocks), device=device, dtype=torch.float32)
    partial_weights_all = []
    residual_blocks_all = []

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

    assert b % model_input_bs == 0, "`b` should be divisible by `model_input_bs`"
    num_batches = b // model_input_bs

    print("Number of batches =", num_batches)

    for i in range(num_blocks):
        # Access the first transformer layer and q_proj weight matrix
        layer = model.model.decoder.layers[i]
        self_attn = layer.self_attn

        weight_matrix = [self_attn.q_proj.weight,
                         self_attn.k_proj.weight,
                         self_attn.v_proj.weight,
                         self_attn.out_proj.weight,
                         layer.fc1.weight,
                         layer.fc2.weight]

        partial_weight_layer = []
        residual_blocks_layer = []
        for j in range(num_layers):
            # for each weight_matrix[i] take 'line_size' parameters from the first row of the 'weight_matrix'
            w = weight_matrix[j]
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
            for i_layer in range(num_blocks):
                def custom_forward(layer_number, matrix_number):
                    def _custom_forward(self, inpt):
                        __from = t * num_layers * layer_number + matrix_number * t
                        __to = __from + t

                        _param_block = x[__from:__to]
                        _residual_block = residual_blocks_all[layer_number][matrix_number]
                        concatenated_line = torch.cat((_param_block.to(device=_residual_block.device), _residual_block),
                                                      dim=0)
                        line_reshaped = concatenated_line.unsqueeze(0)
                        _partial_weight = partial_weights_all[layer_number][matrix_number]
                        full_weights = torch.cat((line_reshaped, _partial_weight), dim=0)

                        return F.linear(inpt, full_weights, self.bias)

                    return _custom_forward

                _layer = model.model.decoder.layers[i_layer]
                _self_attn = _layer.self_attn

                modules = [_self_attn.q_proj,
                           _self_attn.k_proj,
                           _self_attn.v_proj,
                           _self_attn.out_proj,
                           _layer.fc1,
                           _layer.fc2]

                for j_matrix in range(num_layers):
                    modules[j_matrix].forward = custom_forward(i_layer, j_matrix).__get__(modules[j_matrix],
                                                                                          type(modules[j_matrix]))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs)

        return partial_ppl_fn

    hess = torch.zeros((t * num_layers * num_blocks, t * num_layers * num_blocks))

    # NOTICE: here we rely on the additive property of the PPL function (Corollary in a technical report)
    for k in range(num_batches):
        dH = hessian(get_partial_ppl_fn(i_start=k * model_input_bs), params)
        hess = hess.to(device=dH.device)
        hess += dH / num_batches

    torch.save(hess, "data/diff_bs/hessian_all_layers_all_blocks.pt")

    return hess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    check_gpus()

    start_t = time.perf_counter()
    hess = compute_hessian_several_layers_several_blocks(model_name=args.model,
                                                         cache_dir=args.cache_dir,
                                                         seed=args.seed,
                                                         num_blocks=12, num_layers=6, t=25)
    print("Computation time =", time.perf_counter() - start_t)

    # Computation time = 90927.86469955707 (all layers, all blocks, t=25)
    # Computation time = 71593.03141659603 (all layers, 1 block, t=300)


    plot_heatmap(hess)
