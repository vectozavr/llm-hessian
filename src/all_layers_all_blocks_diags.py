import argparse
from utils import *
import time

from hessian_diag_single_layer import compute_hessian_diag_hutchinson

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--vhp_samples", type=int, default=5000)
    parser.add_argument("--b", type=int, default=60)
    parser.add_argument("--model_input_bs", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    args = parser.parse_args()

    check_gpus()

    model, tokenizer = get_llm(args.model, args.cache_dir)
    blocks = get_all_blocks(model)

    for block_index in range(len(blocks)):
        block = blocks[block_index]
        layers = find_layers(block)
        for layer_name in layers:
            print('Computing Hessian for ' + layer_name + ' from block ' + str(block_index) + ' of ' + args.model)

            start_t = time.perf_counter()
            hess_diag = compute_hessian_diag_hutchinson(model_name=args.model, layer_name=layer_name,
                                                        vhp_samples=args.vhp_samples, block_index=block_index,
                                                        b=args.b, model_input_bs=args.model_input_bs, seqlen=args.seqlen,
                                                        seed=args.seed, cache_dir=args.cache_dir)
            print("Computation time =", time.perf_counter() - start_t)

            out_path_prefix = "data/" + args.model + "/diag_hessian/" + str(block_index) + "/" + layer_name + "_vhp_samples" + str(args.vhp_samples) + "_b" + str(args.b) + "_seed" + str(args.seed)
            plot_heatmap(torch.abs(hess_diag), out_path_prefix + '.pdf')
            torch.save(hess_diag, out_path_prefix + ".pt")
