# Hessian of Perplexity for Large Language Models by PyTorch autograd (Open Source)
A GitHub repository with [the Technical report]() on how to accurately compute the Hessian of Perplexity function for Large Language Models (LLMs).

*Ivan Ilin*<br>
GenAI Center of Excellence, King Abdullah University of Science and Technology, Thuwal, Saudi Arabia<br>
[Paper](link)

```bibtex
@article{hessian2025llm,
  title={Hessian of Perplexity for Large Language Models by PyTorch autograd (Open Source)}, 
  author={Ivan Ilin},
  year={2025},
  journal={arXiv preprint arXiv:}
}
```

---

## Results

### Full Hessian (for different subsets of parameters)

<b>On the left</b>: Hessian of Perplexity for q_proj linear layer $Q_{proj} \in \mathbb{R}^{768 \times 768}$ from self attention of the first block of OPT-125M (for the first 768 parameters = first row of q_proj):

<b>On the right</b>: Hessian of Perplexity for all linear layers from the first block of OPT-125M (for the first 1800 parameters = 300 parameters for every linear layer $\times$ 6 linear layers):

<p align="center">
  <img src="pdf/hessian_q_proj_t_768.png" alt="Hessian of q_proj" width="49%"/>
  <img src="pdf/hessian_all_layers_first_block_t_300.png" alt="Hessian of all layers first block" width="49%"/>
</p>

<b>On the left</b>: Hessian of Perplexity for q_proj from all 12 blocks of OPT-125M (for the first 1800 parameters = 150 parameters for every q_proj layer $\times$ 12 blocks):

<b>On the right</b>: Hessian of Perplexity for all linear layers from all 12 blocks of OPT-125M (for the first 1800 parameters = 25 parameters for every linear layer $\times$ 6 linear layers for a block $\times$ 12 blocks):

<p align="center">
  <img src="pdf/hessian_q_proj_all_blocks_t_150.png" alt="Hessian of q_proj for all blocks" width="49%" />
  <img src="pdf/hessian_all_layers_first_block_t_300.png" alt="Hessian of all layers first block" width="49%" />
</p>

Experiments with different batch size $b \in \\{1, \cdots, 140\\}$:

<img src="pdf/losses_vs_bs.png" alt="Different batch size" width="100%"/>


### Diagonals of the Hessian (for the entire linear layer)

Experiments with different number of Vector-Hessian Product samples $k \in {1, \cdots, 3000}$:

<img src="pdf/losses_vs_k.png" alt="Different vhp k" width="100%"/>

## Setup
Python 3.12.4
```sh
pip install -r requirements.txt
```
## Parameters
We provide a quick overview of the arguments:  
- `--model`: The identifier for the model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--seed`: Specifies a seed.
- `--vhp_samples`: Specifies the number of Vector-Hessian Product samples for diagonal of the Hessian estimation.

## Available scripts

* If you want to compute the Hessian of Perplexity <b>for a single linear layer</b> $Q_{proj} \in \mathbb{R}^{768 \times 768}$ of LLM:
```sh 
python single_layer_single_block.py \
    --model facebook/opt-125m \
    --cache_dir llm_weights \
    --seed 0
```

* If you want to compute the Hessian of Perplexity <b>for a particular linear layer from all blocks</b> of LLM
```sh 
python single_layer_several_blocks.py \
    --model facebook/opt-125m \
    --cache_dir llm_weights \
    --seed 0
```

* If you want to compute the Hessian of Perplexity <b>for several linear layers from all several blocks</b> of LLM
```sh 
python several_layers_several_blocks.py \
    --model facebook/opt-125m \
    --cache_dir llm_weights \
    --seed 0
```

* If you want to compute the Diagonal of the Hessian of Perplexity for q_proj layer of the first block of LLM:
```sh 
python q_proj_hessian_diag.py \
    --model facebook/opt-125m \
    --cache_dir llm_weights \
    --seed 0
    --vhp_samples 100
```

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
