from typing import List, Dict
import math
import itertools
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
from tqdm import tqdm


def evaluate_generations(
    generations: List[str],
    *,
    ppl_model_name: str = "gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:

    # mean length
    lengths = [len(t.split()) for t in generations]
    mean_length = sum(lengths) / len(lengths)

    # perplexity
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name).to(device)
    model.eval()

    with torch.no_grad():
        nlls = []
        for text in tqdm(generations, desc="Calculating perplexity", leave=False):
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            # Shift so that tokens predict the *next* token
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            # Cross-entropy over all tokens
            neg_log_likelihood = outputs.loss.item() * (input_ids.size(1) - 1)
            nlls.append(neg_log_likelihood)

    ppl = math.exp(sum(nlls) / sum(lengths))

    # self bleu
    bleu_scores = []
    for idx, hyp in enumerate(tqdm(generations, desc="Calculating self-BLEU", leave=False)):
        refs = generations[:idx] + generations[idx + 1 :]
        bleu = sacrebleu.corpus_bleu([hyp], [refs]).score
        bleu_scores.append(bleu)
    self_bleu = sum(bleu_scores) / len(bleu_scores)

    # distinct n
    def distinct_n(n: int) -> float:
        ngrams = [
            tuple(tokens[i : i + n])
            for tokens in (gen.split() for gen in generations)
            for i in range(len(tokens) - n + 1)
        ]
        return len(set(ngrams)) / max(1, len(ngrams))

    distinct_1 = distinct_n(1)
    distinct_2 = distinct_n(2)

    return {
        "mean_length": mean_length,
        "perplexity": ppl,
        "self_bleu": self_bleu,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
    }