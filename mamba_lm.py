from dataclasses import dataclass, fields, asdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba, MambaConfig, RMSNorm
from safetensors.torch import load_file as load_file_safetensors

"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

# TODO generate function : batch size != 1 ? (for now B=1)
# TODO generate function : top-p sampling

@dataclass
class MambaLMConfig(MambaConfig):
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8

    def __post_init__(self):
        super().__post_init__()

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)

# adapted from https://github.com/johnma2006/mamba-minimal
def from_pretrained(name: str):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.

    Note :
    This only work with the state-spaces/mamba-XXX model family, because there is a pytorch_model.bin file in the HF repo.
    This is not the case of typical model saved on HF (like the state-spaces/mamba-XXX-hf model family).
    To load the state dict of such models, I think the only way is to load the model into a AutoModelForCausalLM, and then
    pass the state_dict to a MambaLM. I see no other way around unfrortunately (this is how it's done in jamba.py)

    Args:
        name: As of now, supports
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'

    Returns:
        model: a Mamba model configured with the proper parameters and initialized with the proper weights
    """   

    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    def load_config_hf(model_name):
        resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        return json.load(open(resolved_archive_file))
                
    def load_state_dict_hf(model_name):
        if model_name.endswith('-hf') or model_name.endswith('-rw'):
            resolved_archive_file = cached_file(model_name, "model.safetensors", _raise_exceptions_for_missing_entries=False)
            print(resolved_archive_file)
            return load_file_safetensors(resolved_archive_file, device='cpu') # torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        else:
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            print(resolved_archive_file)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
    # copy config data
    config_data = load_config_hf(name)
    print(config_data)
    # config = MambaLMConfig(d_model=config_data['d_model'], n_layers=config_data['n_layer'], vocab_size=config_data['vocab_size'])
    config = MambaLMConfig(d_model=(config_data['hidden_size'] if 'hidden_size' in config_data.keys() else config_data['d_model']), n_layers=config_data['n_layer'], vocab_size=config_data['vocab_size'])

    model = MambaLM(config)

    def replace(old_dict, new_dict, old_key, new_key):
        new_dict[new_key] = old_dict[old_key]
        del old_dict[old_key]
    # copy weights
    state_dict = load_state_dict_hf(name)
    if name == "TRI-ML/mamba-7b-rw":
        new_state_dict = {}
        replace(state_dict, new_state_dict, 'embeddings.weight', 'backbone.embedding.weight')
        for i in range(64):
            for continuation in [
                "mixer.A_log",
                "mixer.conv1d.bias",
                "mixer.conv1d.weight",
                "mixer.D",
                "mixer.dt_proj.bias",
                "mixer.dt_proj.weight",
                "mixer.in_proj.weight",
                "mixer.out_proj.weight",
                "mixer.x_proj.weight",
                "norm.weight",
            ]:
                replace(state_dict, new_state_dict, f"layers.{i}.{continuation}", f"backbone.layers.{i}.{continuation}")
        replace(state_dict, new_state_dict, 'norm_f.weight', 'backbone.norm_f.weight')
        replace(state_dict, new_state_dict, 'model.lm_head.weight', 'lm_head.weight')
        state_dict = new_state_dict


    # print(sorted(state_dict.keys()))

    # if 'backbone.embeddings.weight' in state_dict.keys():
    #     state_dict['backbone.embedding.weight'] = state_dict['backbone.embeddings.weight']
    #     del state_dict['backbone.embeddings.weight']
    
    # if "embeddings.weight" in state_dict.keys():
    #     state_dict['backbone.embedding.weight'] = state_dict['embeddings.weight']
    #     del state_dict['embeddings.weight']

    # keys = list(state_dict.keys())
    # for key in keys:
    #     if key != 'model.lm_head.weight' and 'backbone' not in key:
    #         state_dict[f"backbone.{key}"] = state_dict[key]
    #         del state_dict[key]

    # if "lm_head.weight" not in state_dict.keys():
    #     if "model.lm_head.weight" in state_dict.keys():
    #         state_dict["lm_head.weight"] = state_dict["model.lm_head.weight"]
    #         del state_dict["model.lm_head.weight"]
    #     else:
    #         state_dict["lm_head.weight"] = state_dict["backbone.embedding.weight"]


    new_state_dict = {}
    for key in state_dict:
        if key == 'backbone.embedding.weight' or key == 'backbone.norm_f.weight':
            new_key = key.replace('backbone.', '')
        else:
            new_key = key.replace('backbone', 'mamba')

        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    return model

class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig):
        super().__init__()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()

        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x = self.mamba(x)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits

    def forward_with_caches(self, tokens, requested_caches: list[int]):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        # logits : (B) (tells us the last token in each sequence, i.e. the one for which we want the cache)


        x = self.embedding(tokens)

        x, caches = self.mamba.forward_with_caches(x, requested_caches=requested_caches)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits, caches
    
    def step(self, token, caches):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits, caches
    
    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [(None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=input_ids.device)) for _ in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.step(input_ids[:, i], caches) # (batch_size, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                
        outputs = [tokenizer.decode(output.tolist()) for output in input_ids]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs
    