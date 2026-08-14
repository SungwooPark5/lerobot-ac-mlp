from dataclasses import dataclass
                                                                                                                                       
from lerobot.configs.policies import PreTrainedConfig                                                                                    
from lerobot.policies.acm2_icpe_sscp.configuration_acm2_icpe_sscp import ACM2ICPESSCPConfig
                                                                                                                                       
              
@PreTrainedConfig.register_subclass("acm2_bimamba_icpe_sscp")                                                                            
@dataclass      
class ACM2BiMambaICPESSCPConfig(ACM2ICPESSCPConfig):                                                                                     
  """ACM2 + ICPE + SSCP with Bidirectional Mamba2 decoder.
                                                                                                                                       
  Replaces the unidirectional Mamba2 decoder in ACM2ICPE with a BiMamba variant
  (forward + backward stacks of Mamba2 layers, averaged output).                                                                       
                                                                                                                                       
  Inherits all ICPE and SSCP hyperparameters from ACM2ICPESSCPConfig unchanged.
                                                                                                                                       
  Motivation: Mamba2 lacks the partial bidirectional integration that Mamba3
  provides via β_t B_{t-1} x_{t-1}. As a result, BiMamba's incremental                                                                 
  contribution is empirically much larger on Mamba2 than on Mamba3.                                                                    
  Combining BiMamba (the strongest enhancement for Mamba2) with ICPE + SSCP                                                            
  targets all three SSM-specific limitations: bidirectionality, position                                                               
  awareness, and inter-chunk continuity.                                                                                               
  """                                                                                                                                  
  # Override ACM2Config default — this variant ALWAYS uses BiMamba
  use_bimamba_decoder: bool = True