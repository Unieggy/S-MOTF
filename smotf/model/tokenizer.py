"""
Tokenizer: state dict -> H ∈ [B, 5, d].

Each of the 5 modalities has a *different* input size (base 12, legs 24,
contacts 4, command 3, action 12). The transformer needs them all at the same
width d=256, so we give each modality its OWN linear projection into d and stack
the results into 5 tokens.

The action token is special: during flow matching it carries the *noisy* action
`a_noisy` at flow-time `t`, so we add a sinusoidal time embedding to that token
(and only that token) telling the network where it is on the noise->data path.

Stack order is FIXED and load-bearing: [base, legs, contacts, command, action]
(row 0 = base, row 4 = action). The heads in Step 4 slice by these indices.
"""
import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbed(nn.Module):
        """
        Map a scalar flow-time t in [0,1] to a d-dim embedding, then an MLP
        """

        def __init__(self,d):
            super().__init__()
            self.d=d
            #two layer with SiLU swish non linear to map raw frequencies from B,d to B,d
            self.mlp=nn.Sequential(
                    nn.Linear(d,d),
                    nn.SiLU(),
                    nn.Linear(d,d),

            )
        def forward(self,t): #input t shape [B], every batch has a single time scalr [0,1]
              #map B to B,128
              # wi(i=0..127)=e^-(ln(10000)*i)/(d/2)
              half=self.d//2 #half dim for sin, half for cos d//2=128
            
              #1d tensor of shape [128]
              freqs=torch.exp(-math.log(10000.0)*torch.arange(half,device=t.device)/half)

              #multiple every scalr t in the batch to freq
              #t[:,None] becomes [B,1]
              #freqs[None,:] becomes [1,128]
              #becomes B,128
              args=t[:,None]*freqs[None,:]

              #torch.sin: B,128
              #toch.cons :B,128, stack them on dim=-1  results in B,256
              emb=torch.cat([torch.sin(args),torch.cos(args)],dim=-1)

              return self.mlp(emb) 
        
class Tokenizer(nn.Module):
    
      
    #map each of five modalities to 256
    ORDER=["base", "legs", "contacts", "command", "action"]

    def __init__(self,cfg):
        super().__init__()
        d=cfg.d #embed dim
        dims=cfg.dims
        
        #dict of layers
        self.proj=nn.ModuleDict({
                "base": nn.Linear(dims.base,d), # B,12 to B,256
                "legs": nn.Linear(dims.legs,d), #B,24 to B,256
                "contacts": nn.Linear(dims.contacts,d),
                "command": nn.Linear(dims.command,d),
                "action": nn.Linear(dims.action,d),#B,12 to B,256
        })

        self.time_embed=SinusoidalTimeEmbed(d) # instantiate the time encoder obj


    def forward(self,batch,a_noisy,t):
         
         #linear mapping over all 5 distinct modules
         tokens={
              "base": self.proj["base"](batch["base"]), # B,12 * 12,256
              "legs": self.proj["legs"](batch["legs"]),
              "contacts": self.proj["contacts"](batch["contacts"]),
              "command": self.proj["command"](batch["command"]),

              #action conditioning injection: add the time embed
              "action": self.proj["action"](a_noisy)+self.time_embed(t),
         }



         #extracts the tokens, gather 5 tensor of shape [B,256]
         #[B,5,256]
         H=torch.stack([tokens[k]for k in self.ORDER],dim=1)
         return H

if __name__ =="__main__":
     from smotf import load_config

     cfg=load_config()
     tok=Tokenizer(cfg).eval()
     B=4

     batch={
          "base": torch.randn(B,cfg.dims.base),
          "legs": torch.randn(B,cfg.dims.legs),
          "contacts": torch.randn(B,cfg.dims.contacts),
          "command": torch.randn(B,cfg.dims.command),
     }
     a_noisy=torch.randn(B,cfg.dims.action)
     t=torch.rand(B)
     H=tok(batch,a_noisy,t)
     print("H shape:",tuple(H.shape)) # 4,5,256

     #test, compute a secondary token array using alternative randomized time variables

     H2=tok(batch,a_noisy,torch.rand(B))
     rows_0_3_same = torch.allclose(H[:, :4], H2[:, :4])
     row_4_diff = not torch.allclose(H[:, 4], H2[:, 4])
     print("change t  -> rows 0-3 unchanged:", rows_0_3_same, "| row 4 changed:", row_4_diff)

     # --- changing a_noisy must change ONLY row 4 ---
     H3 = tok(batch, torch.randn(B, cfg.dims.action), t)
     rows_0_3_same = torch.allclose(H[:, :4], H3[:, :4])
     row_4_diff = not torch.allclose(H[:, 4], H3[:, 4])
     print("change a  -> rows 0-3 unchanged:", rows_0_3_same, "| row 4 changed:", row_4_diff)


    




              
            
                

