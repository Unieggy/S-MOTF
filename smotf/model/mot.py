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
    def _attention(self,x):
        """x: [B, 5, d] (already LayerNorm'd per token) -> [B, 5, d]."""
        B=x.shape[0]

        #per-token QKV, each module has its own qkv weights
        #x[:,i] grabs the ith token for the whole batch[B,256]
        #maps it to [B,768]
        qkv=torch.stack([self.qkv[i](x[:,i]) for i in range(N_TOKENS)],dim=1)#[B, 5, 768]
        #splits q,k,v into 3 equal matrix of 256
        q,k,v=qkv.chunk(3,dim=-1) #each [B,5,256]

        #reshape for multi head splitting

        #[B,heads,5,head_dim] #heads*headdim=256,only reinterprets the same data
        def to_heads(t):
            return t.view(B,N_TOKENS,self.n_heads,self.head_dim).transpose(1,2)
        
        q,k,v=to_heads(q),to_heads(k),to_heads(v)

        #shared crossattention
        out=F.scaled_dot_product_attention(q,k,v) #[B,heads,5,head_dim]

        #merge heads[B,heads,5,head-dim]to [B,5,d]
        out=out.transpose(1,2).reshape(B,N_TOKENS,self.d)

        #per token projection
        #[B,5,256]
        out=torch.stack([self.proj_out(out[:,i])for i in range(N_TOKENS)],dim=1)
        return out
    
    def forward(self,x): #x[B,5,256]
        #residual connections

        #residual 1, input+attention_output
        #layernorm&attention

        #pass thru layernorm. Shape[B,5,256]
        #apply layernorm for every module with dim [B,256] 
        ln_x=torch.stack([self.ln[i](x[:,i])for i in range(N_TOKENS)],dim=1)

        x=x+self._attention(ln_x)

        #residual 2: ffn

        #shape:[B,5,256]
        ffn_out=torch.stack([self.ffn[i](x[:,i])for i in range(N_TOKENS)],dim=1)

        x=x+ffn_out

        return x
    
class MoTBackbone(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.blocks=nn.ModuleList([MoTBlock(cfg) for _ in range(cfg.n_blocks)])

    
    def forward(self,x):
        for block in self.blocks:
            #forward pass 
            x=block(x)
        return x


if __name__=="__main__":
    from smotf import load_config

    cfg=load_config()
    backbone=MoTBackbone(cfg).eval()
    B=4
    x=torch.randn(B,N_TOKENS,cfg,d)

    y=backbone(x)
    print("shape:", tuple(y.shape))                       # (4, 5, 256)
    # --- decoupling test: zero the LEGS (row 1) FFN of block 0, check only row 1's
    #     FFN contribution changes (experts are not shared) ---
    block = MoTBlock(cfg).eval()
    y_before = block(x)
    with torch.no_grad():
        for p in block.ffn[1].parameters():
            p.zero_()
    y_after = block(x)

    changed = [not torch.allclose(y_before[:, i], y_after[:, i]) for i in range(N_TOKENS)]
    print("rows changed after zeroing legs FFN:", changed)   # only index 1 -> True

    # --- gradient-flow test: every per-token group gets grads ---
    x2 = torch.randn(B, N_TOKENS, cfg.d, requires_grad=False)
    out = backbone(x2).sum()
    out.backward()
    b0 = backbone.blocks[0]
    all_have_grad = all(
        b0.ln[i].weight.grad is not None
        and b0.qkv[i].weight.grad is not None
        and b0.ffn[i][0].weight.grad is not None
        for i in range(N_TOKENS)
    )
    print("all per-token LN/QKV/FFN groups received grads:", all_have_grad)
        






















