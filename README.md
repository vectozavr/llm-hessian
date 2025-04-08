# Hessian of Perplexity for Large Language Models by PyTorch autograd (Open Source)
A GitHub repository with [the Technical report]() on how to accurately compute the Hessian of Perplexity function for Large Language Models (LLMs).

*Ivan Ilin*<br>
GenAI Center of Excellence, King Abdullah University of Science and Technology, Thuwal, Saudi Arabia<br>
[Paper](link)

```bibtex
@misc{ilin2025hessian,
      title={Hessian of Perplexity for Large Language Models by PyTorch autograd (Open Source)}, 
      author={Ivan Ilin},
      year={2025},
      eprint={2504.04520},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04520}, 
}
```

---

## Results

### Full Hessian (for different subsets of parameters)
<p>
  <img src="pdf/hessian_q_proj_t_768.png" alt="Hessian of q_proj" width="49%"/>
  <img src="pdf/hessian_all_layers_first_block_t_300.png" alt="Hessian of all layers first block" width="49%"/>
</p>

<b>On the left</b>: Hessian of Perplexity for q_proj linear layer $Q_{proj} \in \mathbb{R}^{768 \times 768}$ from self attention of the first block of OPT-125M (for the first 768 parameters = first row of q_proj):

<b>On the right</b>: Hessian of Perplexity for all linear layers from the first block of OPT-125M (for the first 1800 parameters = 300 parameters for every linear layer $\times$ 6 linear layers):

<p>
  <img src="pdf/hessian_q_proj_all_blocks_t_150.png" alt="Hessian of q_proj for all blocks" width="49%" />
  <img src="pdf/hessian_all_layers_first_block_t_300.png" alt="Hessian of all layers first block" width="49%" />
</p>

<b>On the left</b>: Hessian of Perplexity for q_proj from all 12 blocks of OPT-125M (for the first 1800 parameters = 150 parameters for every q_proj layer $\times$ 12 blocks):

<b>On the right</b>: Hessian of Perplexity for all linear layers from all 12 blocks of OPT-125M (for the first 1800 parameters = 25 parameters for every linear layer $\times$ 6 linear layers for a block $\times$ 12 blocks):

<img src="pdf/losses_vs_bs.png" alt="Different batch size" width="100%"/>

Experiments with different batch size $b \in \\{1, \cdots, 140\\}$.


### Diagonals of the Hessian (for the entire linear layer)

<img src="pdf/losses_vs_k.png" alt="Different vhp k" width="100%"/>

Experiments with different number of Vector-Hessian Product samples $k \in {1, \cdots, 3000}$.

## Setup
Python 3.12.4
```sh
pip install -r requirements.txt
```
## Parameters
We provide a quick overview of the arguments:  
- `--model`: The identifier for the model on the Hugging Face model hub.
- `--layer_name`: The name of the linear layer for which you want to compute the Hessian.
- `--t`: The number of parameters to consider per linear layer.
- `--block_index`: The index of the block to consider. This is for the case when we consider only one block - `single_layer_single_block.py` or `q_proj_hessian_diag.py`.
- `--num_blocks`: The number of blocks to consider. Applied to `single_layer_several_blocks.py` and `several_layers_several_blocks.py`.
- `--num_layers`: The number of linear layers to consider (per block). Applied to `several_layers_several_blocks.py`. The hessian matrix $H\in \mathbb{R}^{t \cdot num_layers \cdot num_blocks \times t \cdot num_layers \cdot num_blocks}$.
- `--b`: The total number of samples we use to compute the perplexity function. Higher `b` requires more time to finish computations.
- `--vhp_samples`: Specifies the number of Vector-Hessian Product samples for diagonal of the Hessian estimation. Applied to `q_proj_hessian_diag.py`.
- `--model_input_bs`: The number of samples we use at once to compute the perplexity. Higher `model_input_bs` requires more GPU memory.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--seed`: Specifies a seed.

> [!NOTE]  
> Please note that after running the script, a `.pt` Hessian tensor and a `.pdf` heatmap of the Hessian will be saved in the `/data` folder.

> [!WARNING]
> Larger values of `--b` and `--vhp_samples` result in a more accurate representation of the Hessian, but can significantly increase computation time.

> [!TIP]  
> If you have GPUs with large memory capacity, you can try using a larger `--model_input_bs`, which will reduce computation time.

## Run computations

### Single linear layer from a single block
* If you want to compute the Hessian of Perplexity <b>for a single linear layer</b> $Q_{proj} \in \mathbb{R}^{768 \times 768}$ of LLM:
```sh 
python single_layer_single_block.py \
    --model facebook/opt-125m \
    --layer_name self_attn.q_proj \
    --block_index 0 \
    --t 5 \
    --b 30 \
    --model_input_bs 2 \
    --seed 0 \
    --cache_dir llm_weights
```

### Linear layer for several blocks
* If you want to compute the Hessian of Perplexity <b>for a particular linear layer from all blocks</b> of LLM
```sh 
python single_layer_several_blocks.py \
    --model facebook/opt-125m \
    --layer_name self_attn.q_proj \
    --t 5 \
    --num_blocks 3 \
    --b 30 \
    --model_input_bs 2 \
    --seed 0 \
    --cache_dir llm_weights
```

### Several Linear layers for several blocks
* If you want to compute the Hessian of Perplexity <b>for several linear layers from all several blocks</b> of LLM
```sh 
python several_layers_several_blocks.py \
    --model facebook/opt-125m \
    --t 5 \
    --num_layers 3 \
    --num_blocks 3 \
    --b 30 \
    --model_input_bs 2 \
    --seed 0 \
    --cache_dir llm_weights
```

### Estimation of the Full Hessian Diagonal elements
* If you want to compute the Diagonal of the Hessian of Perplexity for q_proj layer of the first block of LLM:
```sh 
python hessian_diag_single_layer.py \
    --model facebook/opt-125m \
    --layer_name self_attn.q_proj \
    --vhp_samples 10 \
    --block_index 0 \
    --b 30 \
    --model_input_bs 2 \
    --seed 0
    --cache_dir llm_weights
```

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
