import argparse
import time
from utils import *

from single_layer_single_block import compute_hessian_single_layer_single_block


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--block_index", type=int, default=0)
    parser.add_argument("--t", type=int, default=512)
    parser.add_argument("--b", type=int, default=60)
    parser.add_argument("--model_input_bs", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()

    model, tokenizer = get_llm(args.model, args.cache_dir)
    block = get_all_blocks(model)[args.block_index]
    layers = find_layers(block)

    # INFO: This file is doing the same as `single_layer_single_block.py` but in cycle for all linear layers from block[block_index]
    for layer_name in layers:
        print('Computing Hessian for ' + layer_name + ' from block ' + str(args.block_index) + ' of ' + args.model)
        start_t = time.perf_counter()
        hess = compute_hessian_single_layer_single_block(model_name=args.model, layer_name=layer_name,
                                                         block_index=args.block_index, t=args.t, b=args.b,
                                                         model_input_bs=args.model_input_bs, seqlen=args.seqlen,
                                                         seed=args.seed, cache_dir=args.cache_dir)
        print("Computation time =", time.perf_counter() - start_t)

        out_path_prefix = "data/llama3-1b/hessian_" + layer_name + "_block" + str(args.block_index) + "_t" + str(args.t) + "_b" + str(args.b) + "_seed" + str(args.seed)
        plot_heatmap(torch.abs(hess), out_path_prefix + '.pdf')
        torch.save(hess, out_path_prefix + ".pt")
