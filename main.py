from model import GPT2NoiseWrapper
from metrics import evaluate_generations

from typing import List, Callable, Any
from contextlib import nullcontext
import torch
import json
from tqdm import tqdm

def batch_generate(
    inputs: List[str],
    model: Callable[..., Any],
    tokenizer: Callable[[str], Any] | None = None,
):


    base_prompt = "write a thoughtful poem:\n"

    outputs: List[str] = []
    with torch.no_grad(): 
        for item in tqdm(inputs, 'generating', leave=False):
            item= base_prompt + item
            prepared = tokenizer(item) if tokenizer else item
            raw_out = model.generate(prepared, max_new_tokens=100)
            text_out = raw_out[0] if raw_out else ""

            outputs.append(text_out[len(base_prompt):])  # remove prompt from output
    return outputs

def progressive_noise_factory(strength=0.05):
    """return a function that adds progrssive noise with a given strength."""
    def noise_fn(hidden, layer_idx):
        std = strength * (layer_idx + 1)
        return torch.randn_like(hidden) * std
    return noise_fn

MODEL_CONFIGS=[
    {"temperature": 1, "logit_noise_std": 0.0, "hidden_noise_fn": None},
    {"temperature": 1.25, "logit_noise_std": 0.0, "hidden_noise_fn": None},
    {"temperature": 1.5, "logit_noise_std": 0.0, "hidden_noise_fn": None},
    {"temperature": 1, "logit_noise_std": 0.5, "hidden_noise_fn": None},
    {"temperature": 1, "logit_noise_std": 1, "hidden_noise_fn": None},
    {"temperature": 1, "logit_noise_std": 1.5, "hidden_noise_fn": None},
    {"temperature": 1, "logit_noise_std": 0.0, "hidden_noise_fn": progressive_noise_factory(0.025)},
    {"temperature": 1, "logit_noise_std": 0.0, "hidden_noise_fn": progressive_noise_factory(0.05)},
    {"temperature": 1, "logit_noise_std": 0.0, "hidden_noise_fn": progressive_noise_factory(0.075)},
]

if __name__ == "__main__":
    with open("whitman.json", "r") as f:
        inputs = json.load(f)
    # inputs = inputs[:100]  #100 to approx results quickly. will do a full run later.

    results=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    
    for config in MODEL_CONFIGS:
        print(f"Running with config: {config}")
        model = GPT2NoiseWrapper(**config)
        generations = batch_generate(inputs, model, device="cuda")
        metrics = evaluate_generations(generations)

        results.append((str(config), generations, metrics))
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
