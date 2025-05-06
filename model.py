import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Callable, Optional, Union
import math


class GPT2NoiseWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        temperature: float = 1.0,
        logit_noise_std: float = 0.0,
        hidden_noise_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.model  = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.tok    = GPT2TokenizerFast.from_pretrained(model_name)
        # occassionally errors without this, not sure why
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.temperature     = temperature
        self.logit_noise_std = logit_noise_std
        self.hidden_noise_fn = hidden_noise_fn

        # transformer offers hook to run custom code between each layer
        if hidden_noise_fn is not None:
            for idx, block in enumerate(self.model.transformer.h):
                block.register_forward_hook(self._make_hidden_hook(idx))


    def set_temperature(self, t: float):
        self.temperature = max(t, 1e-6)

    def set_logit_noise(self, std: float):
        self.logit_noise_std = max(std, 0.0)

    def set_hidden_noise_fn(self, fn: Optional[Callable]):
        self.hidden_noise_fn = fn
        for blk in self.model.transformer.h:
            blk._forward_hooks.clear()
        if fn is not None:
            for idx, blk in enumerate(self.model.transformer.h):
                blk.register_forward_hook(self._make_hidden_hook(idx))


    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[str, list[str]],
        max_new_tokens: int = 50,
        **gen_kwargs,
    ) -> list[str]:

        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = self.tok(prompt, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=1,
            top_k=0,
            pad_token_id=self.tok.pad_token_id,
            **gen_kwargs
        )
        return self.tok.batch_decode(outputs, skip_special_tokens=True)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        out = self.model(input_ids=input_ids, **kwargs)

        if self.logit_noise_std > 0:
            noise = torch.randn_like(out.logits) * self.logit_noise_std
            out.logits = out.logits + noise

        return out

    def _make_hidden_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self.hidden_noise_fn is None:
                return output
            
            # output of each block seems to be a tuple. first element is the hidden states, not sure about the rest, but seem to be important
            if isinstance(output, tuple):
                hidden_states = output[0]
                noise = self.hidden_noise_fn(hidden_states, layer_idx)
                return (hidden_states + noise,) + output[1:]
            else:
                noise = self.hidden_noise_fn(output, layer_idx)
                return output + noise
        return hook

if __name__ == "__main__":
    model = GPT2NoiseWrapper(temperature=1.0)
    print('baseline:')
    print(model.generate("In a distant galaxy,"))

    model=GPT2NoiseWrapper(temperature=1, logit_noise_std=1)
    print('logit noise:')
    print(model.generate("In a distant galaxy,"))

    def progressive_noise(hidden, layer_idx):
        """std dev grows linearly with layer depth."""
        std = 0.05 * (layer_idx + 1)
        return torch.randn_like(hidden) * std
    
    model = GPT2NoiseWrapper(temperature=1, hidden_noise_fn=progressive_noise)
    print('hidden noise:')

    model.set_hidden_noise_fn(progressive_noise)
    print(model.generate("In a distant galaxy,"))