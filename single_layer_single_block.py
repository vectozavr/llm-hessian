import argparse
import time

import torch
import torch.nn.functional as F

from torch.autograd.functional import hessian
from utils import set_seed, get_llm, ppl_function, check_gpus, plot_heatmap, plot_hist
from data import get_cached_wikitext2


def compute_hessian_single_layer_single_block(model_name, cache_dir, seed, t=768, block_number=0, model_input_bs=4, b=140):
    # Setting seeds for reproducibility
    set_seed(seed)

    model, tokenizer = get_llm(model_name, cache_dir)
    device = torch.device("cuda:0")

    # Get the test loader
    testloader = get_cached_wikitext2(tokenizer, model.seqlen, seed=seed)

    # Access the first transformer layer and q_proj weight matrix
    layer = model.model.decoder.layers[block_number]
    self_attn = layer.self_attn
    weight_matrix = self_attn.q_proj.weight

    params = weight_matrix[0][:t].clone().requires_grad_(True)
    residuals = weight_matrix[0][t:].clone()
    partial_weights = weight_matrix[1:]

    samples_in_dataset = testloader.input_ids.numel() // model.seqlen

    b = max(1, min(b, samples_in_dataset))
    print("Total number of samples =", b)

    assert b % model_input_bs == 0, "`b` should be divisible by `model_input_bs`"
    num_batches = b // model_input_bs

    print("Number of batches =", num_batches)

    def get_partial_ppl_fn(i_start):
        def partial_ppl_fn(x):
            # Define custom forward method for q_proj
            def custom_q_proj_forward(self, inpt):
                concatenated_line = torch.cat((x, residuals), dim=0)
                line_reshaped = concatenated_line.unsqueeze(0)
                full_weights = torch.cat((line_reshaped, partial_weights), dim=0)

                return F.linear(input=inpt, weight=full_weights, bias=self.bias)

            # Monkey-patch q_proj's forward method
            self_attn.q_proj.forward = custom_q_proj_forward.__get__(self_attn.q_proj, type(self_attn.q_proj))

            return ppl_function(model, testloader, i_start=i_start, device=device, batch_size=model_input_bs)

        return partial_ppl_fn

    hess = torch.zeros((t, t))

    # NOTICE: here we rely on the additive property of the PPL function (Corollary in a technical report)
    for k in range(num_batches):
        t1 = time.perf_counter()

        dH = hessian(get_partial_ppl_fn(i_start=k * model_input_bs), params)
        hess = hess.to(device=dH.device)
        hess += dH / num_batches

        print("dt =", time.perf_counter() - t1)

    torch.save(hess, "data/hessian_q_proj.pt")

    return hess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="facebook/opt-125m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()

    start_t = time.perf_counter()
    hess = compute_hessian_single_layer_single_block(model_name=args.model, cache_dir=args.cache_dir, seed=args.seed,
                                                     t=100, model_input_bs=4, b=140)
    print("Computation time =", time.perf_counter() - start_t)

    print(torch.diag(hess))

    #plot_heatmap(torch.diag(hess).reshape(1, -1))
    #plot_hist(torch.diag(hess))

    # Computation time = 29338.510749154957 sec

    plot_heatmap(torch.abs(hess))
