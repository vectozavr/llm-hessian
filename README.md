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

## Usage

### Setup
Python 3.12.4
```sh
pip install -r requirements.txt
```
### Available scripts

### Parameters
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--seed`: Specifies a seed.

For [LLaMA-3](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-3.2-1B`:
```sh 
python {script_name} \
    --model meta-llama/Llama-3.2-1B \
    --cache_dir llm_weights \
    --seed 0
```

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.