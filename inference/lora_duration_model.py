"""Dependency-free LoRA on the last four w2v-BERT attention Q/V projections.

Training wrappers export merged, hash-bound patches for the unchanged native
inference graph. No PEFT/Transformers dependency is introduced at inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from w2vbert2_multidomain_model import EXPECTED_CHECKPOINT_SHA256, sha256_file

LAYERS = (20, 21, 22, 23)
ADAPTER_PREFIXES = tuple('adapter_layers.'+str(i)+'.' for i in (8,12,16,20,24))+('pooling.','bottleneck.')
TARGETS = tuple('encoder.encoder.layers.%d.self_attn.linear_%s' % (i, qv) for i in LAYERS for qv in ('q','v'))


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters(): p.requires_grad_(False)
        self.scale = float(alpha)/rank
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_a), self.lora_b)

    def merged_weight(self):
        return self.base.weight.detach()+self.scale*(self.lora_b.detach() @ self.lora_a.detach())


def install_lora(model):
    for name in TARGETS:
        parent, leaf = name.rsplit('.',1)
        module = model.get_submodule(parent)
        base = getattr(module, leaf)
        if not isinstance(base, nn.Linear): raise RuntimeError('LoRA target not native linear: '+name)
        setattr(module, leaf, LoRALinear(base))
    head, lora = [], []
    for name, p in model.named_parameters():
        if name.startswith(ADAPTER_PREFIXES): p.requires_grad_(True); head.append(p)
        elif name.endswith(('.lora_a','.lora_b')): p.requires_grad_(True); lora.append(p)
        else: p.requires_grad_(False)
    return head, lora


def export_merged_patch(model, parent_sha256):
    patch = {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(ADAPTER_PREFIXES)}
    for name in TARGETS:
        patch[name+'.weight'] = model.get_submodule(name).merged_weight().cpu()
    return dict(schema_version=1, base_sha256=EXPECTED_CHECKPOINT_SHA256,
                parent_patch_sha256=parent_sha256, state_dict=patch)


def apply_merged_patch(model, path, expected_sha256):
    if sha256_file(path) != expected_sha256: raise RuntimeError('lora patch checksum mismatch')
    obj = torch.load(str(path), map_location='cpu', weights_only=True)
    if set(obj) != {'schema_version','base_sha256','parent_patch_sha256','state_dict'} or obj['schema_version'] != 1:
        raise RuntimeError('lora patch schema mismatch')
    if obj['base_sha256'] != EXPECTED_CHECKPOINT_SHA256: raise RuntimeError('lora base mismatch')
    state = model.state_dict()
    allowed = {k for k in state if k.startswith(ADAPTER_PREFIXES)} | {n+'.weight' for n in TARGETS}
    if set(obj['state_dict']) != allowed: raise RuntimeError('lora patch tensor allowlist mismatch')
    for k,v in obj['state_dict'].items():
        if not isinstance(v,torch.Tensor) or v.shape!=state[k].shape or v.dtype!=state[k].dtype or not torch.isfinite(v).all():
            raise RuntimeError('lora invalid patch tensor: '+k)
    state.update(obj['state_dict']); model.load_state_dict(state,strict=True)
    return obj
