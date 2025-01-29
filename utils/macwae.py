import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from itertools import chain, combinations
from typing import Union
from copy import deepcopy

from pydantic.dataclasses import dataclass
from dataclasses import field

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP

from multivae.data.datasets import MultimodalBaseDataset
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.joint_models import BaseJointModel, BaseJointModelConfig
from multivae.models.nn.default_architectures import BaseDictDecoders, MultipleHeadJointEncoder


@dataclass
class MACWAEConfig(BaseJointModelConfig):
    beta: float = 1.0
    random_init: bool = False
    kernel_bandwidth: float = 1.0
    use_wasserstein: bool = True
    dropout_rate: float = 0.5
    num_dropouts: int = 10
    warmup: int = 10
    text_modalities: list = field(default_factory=lambda: [])
    exhaustive: bool = False
    deterministic: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.num_dropouts is None:
            self.num_dropouts = len(self.input_dims)


def generate_subsets(input_list):
    """
    Generate all possible non-empty subsets of a list of strings.

    Args:
        input_list (list of str): List of strings.

    Returns:
        list of list of str: List of all non-empty subsets.
    """
    return [list(subset) for subset in chain.from_iterable(combinations(input_list, r) for r in range(1, len(input_list) + 1))]


def generate_random_subsets(input_list, p, N):
    """
    Generate N non-empty subsets of the input list with elements omitted with probability p.

    Args:
        input_list (list of str): List of strings.
        p (float): Probability of omitting an element from a subset.
        N (int): Number of subsets to generate.

    Returns:
        list of list of str: List of N randomly generated non-empty subsets.
    """
    subsets = []
    while len(subsets) < N:
        subset = [item for item in input_list if random.random() > p]
        if subset:  # Ensure the subset is non-empty           
            subsets.append(subset)
    return subsets

        
class MACWAE(BaseJointModel):
    def __init__(
        self,
        model_config: MACWAEConfig,
        encoders: dict = None,
        decoders: dict = None,              
        joint_encoder: Union[BaseEncoder, None] = None,
        conditioner: Union[BaseEncoder, None] = None,
        **kwargs,
    ):    

        # Parent init
        super().__init__(
            model_config = model_config, 
            encoders = encoders,
            decoders = decoders, 
            joint_encoder = joint_encoder, 
            **kwargs
        )
                
        # Set model config
        self.model_config = model_config
                    
        # Create conditioner
        if conditioner is None:            
            conditioner = self.default_conditioner(model_config)
        else:
            self.model_config.custom_architectures.append("conditioner")

        self.set_conditioner(conditioner)
        
                                                        
        # Set model name                
        if self.model_config.use_wasserstein:
            self.model_name = 'MACWAE'  
        else:
            self.model_name = 'MACVAE' 
            self.model_config.deterministic = False
        
        # Remove encoders from the model to
        # avoid issues serializing the model 
        delattr(self, 'encoders')
        self.model_config.custom_architectures = [
            archi for archi in self.model_config.custom_architectures 
            if archi != 'encoders'
        ]                
                
    def default_conditioner(self, model_config):
        
        encoders = {
            modality: deepcopy(encoder) 
            for modality, encoder in self.encoders.items()
        }
        
        config = BaseAEConfig(
            input_dim=(model_config.n_modalities,), 
            latent_dim=model_config.latent_dim
        )
        
        encoders['mask'] = Encoder_VAE_MLP(config)
        
        return MultipleHeadJointEncoder(encoders, model_config)
    
    def set_conditioner(self, conditioner):
        "Checks that the provided joint encoder is an instance of BaseEncoder."

        if not issubclass(type(conditioner), BaseEncoder):
            raise AttributeError(
                (
                    f"The joint encoder must inherit from BaseEncoder class from "
                    "multivae.models.nn.default_architectures.BaseJointEncoder . Refer to documentation."
                )
            )
        self.conditioner = conditioner        

    def create_empty_samples(self, n_samples: int):
        
        # Init model output
        output = ModelOutput()
        
        # Create random samples across modalities
        for mod, dims in self.input_dims.items():
            
            if mod in self.model_config.text_modalities:
                
                token_dims = torch.Size((n_samples,) + (dims[0],))
                
                output[mod] = {
                    'tokens' : torch.ones(token_dims, device = self.device).int(),
                    'padding_mask' : torch.zeros(token_dims, device = self.device),                    
                }
                
                output[mod]['tokens'][:,0] = 0
                output[mod]['padding_mask'][:,0] = 1.
                
            else:
                    
                dims = torch.Size((n_samples,) + dims)
                
                output[mod] = torch.zeros(dims, device = self.device)
            
        return output
    
    def create_random_samples(self, n_samples: int):
        
        # Init model output
        output = ModelOutput()
        
        # Create random samples across modalities
        for mod, dims in self.input_dims.items():
            
            if mod in self.model_config.text_modalities:
                
                token_dims = torch.Size((n_samples,) + (dims[0],))
                
                output[mod] = {
                    'tokens' : torch.randint(0, dims[1], token_dims, device = self.device),
                    'padding_mask' : torch.ones(token_dims, device = self.device),                    
                }
                
            else:            
                dims = torch.Size((n_samples,) + dims)
                output[mod] = torch.rand(dims, device = self.device)
            
        return output     
           
        
    def partition_data(self, inputs: MultimodalBaseDataset, cond_mod: list, n_samples = int):
                    
        # Create random samples either by random samples, 
        # or by sampling from the prior
        if self.model_config.random_init:
            output = self.create_random_samples(n_samples)            
        else:
            output = self.create_empty_samples(n_samples)
             
        for mod in cond_mod:
            output[mod] = inputs.data[mod]
            
        # Add mask to partiotioned data
        output['mask'] = (
            torch
            .tensor([mod in cond_mod for mod in self.modalities_name], device = self.device)
            .float()
            .repeat(n_samples, 1)
        )

        output = ModelOutput(output)
        return output
      
    def mini_encode(self, data: dict, sample_shape = [], use_joint_encoder = False):
        if use_joint_encoder:
            output = self.joint_encoder(data)
        else:
            output = self.conditioner(data)
        
        
        if self.model_config.deterministic:
            z = output.embedding
        else:
            mu, logvar = output.embedding, output.log_covariance
            z = dist.Normal(mu, torch.exp(0.5 * logvar)).rsample(sample_shape)
                        
        return ModelOutput(z=z, one_latent_space=True)        
                                
    def encode(self, inputs: MultimodalBaseDataset, cond_mod: Union[list, str] = "all", N: int = 1, **kwargs) -> ModelOutput:
        
        # Set to eval mode
        self.eval()
        
        # If the input cond_mod is a string : convert it to a list
        if type(cond_mod) == str:
            if cond_mod == "all":
                cond_mod = list(self.decoders.keys())
                
            elif cond_mod in self.decoders.keys():
                cond_mod = [cond_mod]
                
            else:
                raise AttributeError(
                    'If cond_mod is a string, it must either be "all" or a modality name'
                    f" The provided string {cond_mod} is neither."
                )
                
        # Get inputs shape
        sample_shape = [] if N == 1 else [N]                
                    
        # Number of samples                
        if cond_mod[0] in self.model_config.text_modalities:
            n_samples = len(inputs.data[cond_mod[0]]['tokens'])
        else:
            n_samples = len(inputs.data[cond_mod[0]])
        
        # Create partial data (by filling missing modalities)
        data = self.partition_data(inputs, cond_mod, n_samples)

        # Encode partial data            
        latent = self.mini_encode(
            data, 
            sample_shape,
            use_joint_encoder=False# use_joint_encoder
        )
            
        return latent

    def get_modality_subsets(self):

        if self.model_config.exhaustive:
            return generate_subsets(
                input_list=self.modalities_name
            )
        else:
            return generate_random_subsets(
                input_list = self.modalities_name, 
                p = self.model_config.dropout_rate, 
                N = self.model_config.num_dropouts
            )
                                
    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:

        # Extract epoch number
        epoch = kwargs.pop("epoch", 1)

        # Number of samples        
        n_samples = len(next(iter(inputs.data.values())))

        # Create subset of modalties
        modality_subsets = self.get_modality_subsets()

        # Aggregate losses across subsets
        reg_loss = 0.
        recon_loss = 0.      
        
        # Iterate over the subsets of modalities                         
        for cond_mod in modality_subsets:
                
            # Create partial data (by filling missing modalities)
            pdata = self.partition_data(inputs, cond_mod, n_samples)
                                        
            # Encode partial data
            cond_out = self.conditioner(pdata)
            
            # Take latent code
            if self.model_config.deterministic:
                z_joint = cond_out.embedding
            else:
                mu, logvar = cond_out.embedding, cond_out.log_covariance 
                z_joint = dist.Normal(mu, torch.exp(0.5 * logvar)).rsample()

            # Joint regularization loss
            if self.model_config.use_wasserstein:
                reg_loss += self.mmd_loss(z_joint)
            else:
                reg_loss += self.kld_loss(mu, logvar)
                            
                                                            
            for mod, decoder in self.decoders.items():             
                x_mod = inputs.data[mod]    
                recon = decoder(z_joint).reconstruction
                # recon = decoder(cond_z_joint).reconstruction
                
                # Modality-specific reconstruction loss 
                if self.model_config.use_wasserstein:
                    recon_loss_mod = self.calc_recon_loss(recon, x_mod)
                else:
                    recon_loss_mod = (
                        -self.recon_log_probs[mod](recon, x_mod) * self.rescale_factors[mod]
                    )
                            
                # Add to reconstruction loss
                recon_loss += recon_loss_mod.sum() / len(x_mod)
                                                                                    
        reg_loss /= self.model_config.num_dropouts
        recon_loss /= self.model_config.num_dropouts               
        
        reg_match_loss = reg_loss * self.model_config.beta 
        if epoch >= self.model_config.warmup:
            annealing_factor = 1
        else:
            annealing_factor = epoch / self.model_config.warmup        
        
        loss = recon_loss + annealing_factor * reg_match_loss
        
        return ModelOutput(
            loss=loss, 
            recon_loss = recon_loss, 
            reg_loss = reg_loss, 
            metrics=dict()
        )

    def calc_recon_loss(self, input: torch.Tensor, target: Union[torch.Tensor, dict]):
        
        if isinstance(target, dict):
            # converts to tokens proba instead of class id for text
            _target = torch.zeros(input.shape[0] * input.shape[1], input.shape[-1]).to(
                input.device
            )
            _target = _target.scatter(1, target["tokens"].reshape(-1, 1), 1)
            input = input.reshape(input.shape[0] * input.shape[1], -1)
        else:
            _target = target.reshape(target.shape[0], -1)
            input = input.reshape(input.shape[0], -1)

        loss = F.mse_loss(input, _target, reduction = 'none')
        return loss

    def gnll_loss(self, z: torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor):
        return F.gaussian_nll_loss(z, mu, logvar.exp(), reduction = 'mean')  # This is better reduction

    def kld_gaussians(self, mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor):
        # Convert log-variances to variances
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        # Calculate the KL divergence
        kl_div = 0.5 * (
            (logvar2 - logvar1) +                 
            ((var1 + (mu1 - mu2)**2) / var2) - 1  
        )
        
        return kl_div.sum() / len(mu1)         

    def kld_loss(self, mu: torch.Tensor, logvar: torch.Tensor):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

    def mmd_loss(self, z: torch.Tensor):
        # Size of the latent batch
        n = z.shape[0]
        
        # Init prior
        # z_prior = self.generate_from_prior(n).z
        z_prior = torch.randn_like(z, device = z.device)
        
        # Kernels
        k_z = self.rbf_kernel(z, z)
        k_z_prior = self.rbf_kernel(z_prior, z_prior)
        k_cross = self.rbf_kernel(z, z_prior)
        
        # Loss comp.
        mmd_z = (k_z - k_z.diag().diag()).sum() / ((n - 1) * n)
        mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((n - 1) * n)
        mmd_cross = k_cross.sum() / (n**2)
        
        return mmd_z + mmd_z_prior - 2 * mmd_cross
 
    def rbf_kernel(self, z1, z2):        
        C = 2.0 * self.model_config.latent_dim * self.model_config.kernel_bandwidth**2
        k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)

        return k
   
    def default_decoders(self, model_config) -> nn.ModuleDict:
        return BaseDictDecoders(
            input_dims=model_config.input_dims,
            latent_dim=model_config.latent_dim,
        )  
