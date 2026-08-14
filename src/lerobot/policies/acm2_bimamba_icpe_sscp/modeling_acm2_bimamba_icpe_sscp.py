"""ACM2 + ICPE + SSCP + Bidirectional Mamba2 (BiMamba)
                                                                                                                                       
Replaces Mamba2ICPEDecoder with a BiMamba variant that:
1. Forward pass:  process [encoder_out, carry?, action_queries] left-to-right                                                          
2. Backward pass: flip the sequence → process → flip back (right-to-left)
3. Average:       0.5 * (forward + backward)                                                                                           
              
The carry token is included in BOTH directions so both forward and backward SSMs                                                         
warm up from previous-chunk context. The last K positions are still extracted                                                            
after averaging.                                                                                                                         
                                                                                                                                       
All SSCP machinery (carry tracking, chunk-pair training) is inherited from                                                               
ACM2ICPESSCPPolicy and works without modification since the model interface
is identical (same forward signature and return shapes).                                                                                 

Mirrors ACM3's `acm3_bimamba_icpe_sscp` module, but uses Mamba2 instead of                                                               
Mamba3 and omits Mamba3-specific arguments (RoPE, MIMO, etc.).                                                                           
"""                                                                                                                                      
                                                                                                                                       
from itertools import chain                                                                                                              
              
import torch
from torch import Tensor, nn
                                                                                                                                       
try:
  from mamba_ssm import Mamba2                                                                                                         
  HAS_MAMBA2 = True
except ImportError:
  HAS_MAMBA2 = False

from lerobot.policies.acm2_bimamba_icpe_sscp.configuration_acm2_bimamba_icpe_sscp import (                                               
  ACM2BiMambaICPESSCPConfig,
)                                                                                                                                        
from lerobot.policies.acm2_icpe.modeling_acm2_icpe import ACM2ICPE
from lerobot.policies.acm2_icpe_sscp.modeling_acm2_icpe_sscp import ACM2ICPESSCPPolicy                                                   
                                                                                                                                       

class Mamba2BiMambaICPEDecoder(nn.Module):                                                                                               
  """BiMamba decoder with carry support, drop-in for Mamba2ICPEDecoder.
                                                                                                                                       
  Accepts the same forward interface as Mamba2ICPEDecoder (x already has
  pos+ICPE baked in). The carry token (B, 1, D) is prepended to both forward                                                           
  and backward sequences so both SSMs warm up from previous-chunk context
  rather than h=0.                                                                                                                     
  """         
                                                                                                                                       
  def __init__(self, config: ACM2BiMambaICPESSCPConfig):
      super().__init__()
      if not HAS_MAMBA2:
          raise ImportError(                                                                                                           
              "mamba-ssm >= 2.0 is required for Mamba2. "
              "Install with: pip install mamba-ssm>=2.0"                                                                               
          )                                                                                                                            

      # offset keeps forward/backward layer_idx disjoint: fwd uses 0..N-1,                                                             
      # bwd uses N..2N-1 — avoids overlapping Mamba2 internal cache state.
      def _make_layers(offset: int = 0):                                                                                               
          return nn.ModuleList([
              Mamba2(                                                                                                                  
                  d_model=config.dim_model,                                                                                            
                  d_state=config.mamba2_d_state,
                  d_conv=config.mamba2_d_conv,                                                                                         
                  expand=config.mamba2_expand,
                  headdim=config.mamba2_headdim,                                                                                       
                  chunk_size=config.mamba2_chunk_size,
                  layer_idx=i + offset,                                                                                                
              )                                                                                                                        
              for i in range(config.n_decoder_layers)
          ])                                                                                                                           
              
      self.forward_layers = _make_layers(0)
      self.backward_layers = _make_layers(config.n_decoder_layers)
      self.norm = nn.LayerNorm(config.dim_model)                                                                                       
      # "pre_query": carry adjacent to the queries in the forward pass (recommended);
      # "prefix": carry at the very front. In BOTH cases the backward pass places                                                      
      # carry immediately before the (reversed) queries so it warms them up — the
      # backward SSM processes queries first, so there is no "prefix" alternative.                                                     
      self.carry_position = getattr(config, "sscp_carry_position", "pre_query")
                                                                                                                                       
  def forward(
      self,                                                                                                                            
      x: Tensor,                       # (K, B, D) — decoder queries (pos+ICPE baked in)
      encoder_out: Tensor,             # (T, B, D) — encoder context (enc_pos baked in)                                                
      decoder_pos_embed: Tensor | None = None,
      encoder_pos_embed: Tensor | None = None,                                                                                         
      carry: Tensor | None = None,     # (B, 1, D) — carry from previous chunk
  ) -> Tensor:                         # returns (K, B, D)                                                                             
      # Parent ACM2ICPE bakes pos_embed into x BEFORE calling decoder, so the
      # decoder_pos_embed arg is None here. We keep the signature for compatibility                                                    
      # with Mamba2ICPEDecoder (which accepts these args).                                                                             
      if decoder_pos_embed is not None:                                                                                                
          x = x + decoder_pos_embed                                                                                                    
      if encoder_pos_embed is not None:                                                                                                
          encoder_out = encoder_out + encoder_pos_embed
                                                                                                                                       
      x = x.transpose(0, 1)                       # (B, K, D)
      encoder_out = encoder_out.transpose(0, 1)   # (B, T, D)                                                                          
                                                                                                                                       
      T = encoder_out.shape[1]
      base = torch.cat([encoder_out, x], dim=1)   # (B, T+K, D)                                                                        
              
      # ── Forward direction ────────────────────────────────────────────                                                              
      if carry is not None:
          if self.carry_position == "prefix":                                                                                          
              fwd = torch.cat([carry, base], dim=1)              # [carry, enc, x]
          else:  # pre_query: [enc, carry, x]                                                                                          
              fwd = torch.cat([encoder_out, carry, x], dim=1)
      else:                                                                                                                            
          fwd = base
      for layer in self.forward_layers:                                                                                                
          fwd = layer(fwd)
      # Drop carry slot so output aligns with base length (T + K)
      if carry is not None:                                                                                                            
          if self.carry_position == "prefix":
              fwd = fwd[:, 1:, :]                                # drop carry@0                                                        
          else:                                                                                                                        
              fwd = torch.cat([fwd[:, :T, :], fwd[:, T + 1:, :]], dim=1)  # drop carry@T
                                                                                                                                       
      # ── Backward direction ───────────────────────────────────────────
      # Flip base, then prepend carry so the backward SSM warms up from                                                                
      # carry immediately before the (reversed) queries. Flipping a                                                                    
      # carry-prefixed sequence would push carry to the end and skip warm-up.
      bwd = base.flip(dims=[1])                                                                                                        
      bwd = torch.cat([carry, bwd], dim=1) if carry is not None else bwd
      for layer in self.backward_layers:                                                                                               
          bwd = layer(bwd)                                                                                                             
      if carry is not None:
          bwd = bwd[:, 1:, :]                                    # drop carry slot                                                     
      bwd = bwd.flip(dims=[1])                                                                                                         

      # ── Average — maintains activation scale vs unidirectional baseline                                                             
      combined = 0.5 * (fwd + bwd)
                                                                                                                                       
      K = x.shape[1]
      out = combined[:, -K:, :]
      out = self.norm(out)
      return out.transpose(0, 1)                                 # (K, B, D)
                                                                                                                                       

class ACM2BiMambaICPESSCP(ACM2ICPE):                                                                                                     
  """ACM2ICPE with BiMamba decoder. Inherits _encode(), forward(), etc.
                                                                                                                                       
  Only changes:
    - self.decoder is Mamba2BiMambaICPEDecoder (forward + backward stacks)                                                             
    - _reset_parameters excludes both layer stacks from Xavier init                                                                    
  """
                                                                                                                                       
  def __init__(self, config: ACM2BiMambaICPESSCPConfig):
      super().__init__(config)                                                                                                         
      # Replace unidirectional decoder with BiMamba decoder
      self.decoder = Mamba2BiMambaICPEDecoder(config)                                                                                  
      self._reset_parameters()
                                                                                                                                       
  def _reset_parameters(self):
      # Handle two layer stacks instead of single self.decoder.layers
      if not hasattr(self.decoder, "forward_layers"):                                                                                  
          # Called during super().__init__() before decoder is replaced — skip
          return                                                                                                                       
      mamba_ids = (
          {id(p) for p in self.decoder.forward_layers.parameters()} |                                                                  
          {id(p) for p in self.decoder.backward_layers.parameters()}
      )                                                                                                                                
      icpe_ids = {id(p) for p in self.icpe_proj.parameters()}
      for p in chain(self.encoder.parameters(), self.decoder.parameters()):                                                            
          if p.dim() > 1 and id(p) not in mamba_ids and id(p) not in icpe_ids:                                                         
              nn.init.xavier_uniform_(p)
                                                                                                                                       
              
class ACM2BiMambaICPESSCPPolicy(ACM2ICPESSCPPolicy):                                                                                     
  """SSCP policy with BiMamba decoder.
                                                                                                                                       
  Inherits all carry tracking and chunk-pair training logic from
  ACM2ICPESSCPPolicy. Only self.model is replaced.                                                                                     
  """         
                                                                                                                                       
  config_class = ACM2BiMambaICPESSCPConfig
  name = "acm2_bimamba_icpe_sscp"                                                                                                      
              
  def __init__(self, config: ACM2BiMambaICPESSCPConfig):                                                                               
      super().__init__(config)
      # Replace model created by parent (ACM2ICPE) with BiMamba version                                                                
      self.model = ACM2BiMambaICPESSCP(config)