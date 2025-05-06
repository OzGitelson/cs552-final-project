"""display most fluent responses"""

import json
from typing import List, Dict
import math
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from compare_results import load_results, choose_condition

def get_most_fluent(idx, data, num_responses=10):

    data=data[idx][1]

    ppl_model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name).to(device)
    model.eval()

    perplexities = []
    with torch.no_grad():
        for text in tqdm(data):
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            neg_log_likelihood = outputs.loss.item() * (input_ids.size(1) - 1)
            perplexity= math.exp(neg_log_likelihood / (input_ids.size(1) - 1))

            perplexities.append(perplexity)


    sorted_data = [x for _, x in sorted(zip(perplexities, data))][-num_responses:]
    print(f"\nMost fluent responses (lowest perplexity):")
    for i, text in enumerate(sorted_data):
        print(f"Response {i+1}:")
        print(text, '\n')

if __name__ == "__main__":
    results=load_results()
    idx = choose_condition(results)
    get_most_fluent(idx, results)