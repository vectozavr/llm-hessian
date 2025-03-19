import random
import torch
from datasets import load_dataset


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_cached_wikitext2(tokenizer, seqlen, seed=0, nsamples=128):
    testloader = None
    try:
        testloader = torch.load("data/wikitext2_dataset_cache.pt")
    except:
        _, testloader = get_wikitext2(nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
        torch.save(testloader, "data/wikitext2_dataset_cache.pt")

    return testloader
