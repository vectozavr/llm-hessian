import torch.nn.functional as F
import argparse
import torch
import time

from torch.autograd.functional import hessian
from utils import set_seed, get_llm, ppl_function, check_gpus, plot_heatmap
from data import get_cached_wikitext2


def compute_hessian_single_layer_several_blocks(model_name, cache_dir, seed, num_blocks=12, t=150, model_input_bs=4, b=140):
    # Setting seeds for reproducibility
    set_seed(seed)

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    params = torch.zeros((t * num_blocks), device=device)
    partial_weights = []
    residuals_blocks = []

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

    assert b % model_input_bs == 0, "`b` should be divisible by `model_input_bs`"
    num_batches = b // model_input_bs

    print("Number of batches =", num_batches)

    for i in range(num_blocks):
        layer = model.model.decoder.layers[i]
        self_attn = layer.self_attn
        weight_matrix = self_attn.q_proj.weight

        # take 'line_size' parameters from the first row of the 'weight_matrix'
        params_in_block = weight_matrix[0][:t].clone().requires_grad_(True)
        residuals = weight_matrix[0][t:].clone()
        partial_weight = weight_matrix[1:]

        params[i * t:(i + 1) * t] = params_in_block

        partial_weights.append(partial_weight)
        residuals_blocks.append(residuals)


    def get_partial_ppl_fn(i_start):
        def partial_ppl_fn(_param_blocks):
            for i_layer in range(num_blocks):
                # Define custom forward method for q_proj
                def custom_q_proj_forward(_i):
                    def _custom_q_proj_forward(self, inpt):
                        _param_block = _param_blocks[_i * t:(_i + 1) * t]
                        _residual_block = residuals_blocks[_i]
                        concatenated_line = torch.cat((_param_block.to(device=_residual_block.device), _residual_block),
                                                      dim=0)
                        line_reshaped = concatenated_line.unsqueeze(0)
                        _partial_weight = partial_weights[_i]
                        full_weights = torch.cat((line_reshaped, _partial_weight), dim=0)

                        return F.linear(inpt, full_weights, self.bias)

                    return _custom_q_proj_forward

                # Monkey-patch q_proj's forward method
                _layer = model.model.decoder.layers[i_layer]
                _self_attn = _layer.self_attn
                _self_attn.q_proj.forward = custom_q_proj_forward(i_layer).__get__(_self_attn.q_proj,
                                                                                   type(_self_attn.q_proj))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs)

        return partial_ppl_fn

    hess = torch.zeros((t*num_blocks, t*num_blocks))

    # NOTICE: here we rely on the additive property of the PPL function (Corollary in a technical report)
    for k in range(num_batches):
        dH = hessian(get_partial_ppl_fn(i_start=k * model_input_bs), params)
        hess = hess.to(device=dH.device)
        hess += dH / num_batches

    return hess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    args = parser.parse_args()

    check_gpus()

    start_t = time.perf_counter()
    hess = compute_hessian_single_layer_several_blocks(model_name=args.model, cache_dir=args.cache_dir, seed=args.seed)
    print("Computation time =", time.perf_counter() - start_t)

    # Computation time = 71551.41185522405 sec

    plot_heatmap(hess)
    torch.save(hess, "data/hessian_q_proj_all_blocks.pt")

