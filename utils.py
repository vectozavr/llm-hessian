import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from torch import nn
from importlib.metadata import version
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer, AutoModelForCausalLM


def draw_progress_bar(current, total, bar_length=30, prefix="Progress"):
    percent = 100 * current / total
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    end_char = '\n' if current == total else ''
    print(f"\r{prefix}: |{bar}| {percent:.2f}%", end=end_char, flush=True)


def check_gpus():
    print("CUDA Available:", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = model.config.max_position_embeddings
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    return model, tokenizer


def get_all_blocks(model):
    if "opt" not in model.name_or_path:
        return model.model.layers
    else:
        return model.model.decoder.layers


def find_layers(block, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        block (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """

    if type(block) in layers:
        return {name: block}
    res = {}

    for name1, child in block.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def val_seqlen(seqlen, model_seqlen):
    if seqlen == 'max':
        seqlen = model_seqlen
    else:
        try:
            seqlen = int(seqlen)
        except:
            print('`seqlen` parameter should be either `max` or convertable to an integer, got `' + seqlen + '`, set seqlen = model.seqlen instead...')
            seqlen = model_seqlen
    return seqlen

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
# Give the average PPL score on a batch of size batch_size
def ppl_function(model, testloader, i_start=0, batch_size=4, device=None, debug=True, seqlen='max'):
    seqlen = val_seqlen(seqlen, model.seqlen)

    # Get input IDs
    testenc = testloader.input_ids
    samples_in_dataset = testenc.numel() // seqlen

    if debug:
        print(f"batch_size = {batch_size}")

    # Calculate end index
    j = min(i_start+batch_size, i_start+samples_in_dataset)

    # Prepare inputs and move to device
    inputs = testenc[:, (i_start * seqlen):(j * seqlen)].to(device)
    inputs = inputs.reshape(j-i_start, seqlen)

    # Forward pass through the model
    lm_logits = model(inputs).logits

    # Shift logits and labels for next token prediction
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = inputs[:, 1:]

    # Compute loss
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    # Calculate negative log likelihood
    neg_log_likelihood = loss.float()

    # IMPORTANT: I skip this step to make this function having additive properties (Corollary 7.2. in a technical report)
    # Compute perplexity
    # ppl = torch.exp(neg_log_likelihood / (nsamples * seqlen))

    return neg_log_likelihood


def model_output_function(model, testloader, i_start=0, batch_size=4, device=None, debug=True, seqlen='max'):
    seqlen = val_seqlen(seqlen, model.seqlen)

    # Get input IDs
    testenc = testloader.input_ids
    samples_in_dataset = testenc.numel() // seqlen

    if debug:
        print(f"batch_size = {batch_size}")

    # Calculate end index
    j = min(i_start+batch_size, i_start+samples_in_dataset)

    # Prepare inputs and move to device
    inputs = testenc[:, (i_start * seqlen):(j * seqlen)].to(device)
    inputs = inputs.reshape(j-i_start, seqlen)

    # Forward pass through the model
    lm_logits = model(inputs).logits

    return lm_logits


def get_nested_attr(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj


def disable_non_differential_modules():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


def plot_heatmap(tensor, out_path="heatmap_hessian.pdf"):
    vmin, vmax = np.percentile(tensor.cpu().numpy(), [0, 98])

    m, n = tensor.shape

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(tensor.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, n, 0, m])

    # Create an axis on the right for the color bar, and match the heat map size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
    plt.colorbar(im, cax=cax, label='Values')

    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    plt.tight_layout(pad=0)

    plt.savefig(out_path, format='pdf', dpi=100)
    plt.close()


def plot_hist(tensor, out_path="histogram.pdf"):
    # Create a histogram-like bar plot
    plt.bar(torch.arange(len(tensor)), tensor.cpu().numpy())

    # Label axes
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.savefig(out_path, format="pdf", bbox_inches="tight")

    # Show plot
    plt.show()
