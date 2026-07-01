"""
Mixture-of-Transformers block.

The core idea: every modality (token) keeps its OWN LayerNorm, QKV projection,
and FFN expert. Only the ATTENTION matmul is shared, so the 5 tokens can
cross-talk while each modality preserves its own magnitude/semantics.

Per token i, one block does:
    x_i = x_i + attn( ln_i(x_i) )        # shared attention over the 5 tokens
    x_i = x_i + ffn_i( x_i )             # per-modality FFN expert

Attention is FULL (no mask) — with only 5 tokens, every token sees every token.
MoTBackbone stacks n_blocks of these.  [B, 5, d] -> [B, 5, d].

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TOKENS=5

class MoTBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.d=cfg.d #hidden dimension 256
        self.n_heads=cfg.n_heads # number of attention heads

        #we must split the 256 dim evenly across heads
        assert self.d%self.n_heads==0,"d must be divisible by n_heads" 
        self.head_dim=self.d//self.n_heads
        
        #5 layer norms for 5 modalities
        self.ln=nn.ModuleList([nn.LayerNorm(self.d) for _ in range(N_TOKENS)])

        #5 qkv projection layers maps 256 to 768
        self.qkv=nn.ModuleList([nn.LayerNorm(self.d,3*self.d) for _ in range(N_TOKENS)])

        #output projections: maps the attention output 256 to 256. 5 separate layers
        self.proj_out=nn.ModuleList([nn.Linear(self.d,self.d) for _ in range(N_TOKENS)])

        #ffn, maps 256, 1024, applies activation, and then shrinks back to 256
        self.ffn=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d,4*self.d),
                nn.GELU(),
                nn.Linear(4*self.d,self.d),
            )
            for _ in range(N_TOKENS)
        ])

        






















