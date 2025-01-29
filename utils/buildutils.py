from multivae.models import (
    JMVAE, 
    JMVAEConfig, 
    MoPoE, 
    MoPoEConfig, 
    MVTCAE,
    MVTCAEConfig
)

from utils.macvae import MACWAE, MACWAEConfig
from utils.nexus_model import Nexus
from utils.nexus_config import NexusConfig

def build_model(
        method: str, 
        params: dict, 
        encoders: dict = None, 
        decoders: dict = None, 
        joint_encoder = None, 
        text_modalities = [], 
        device = 'cpu'
    ):    

    if method == 'MACWAE':
        config = MACWAEConfig(
            n_modalities=params['N_MODALITIES'],
            latent_dim=params['LATENT_DIM'],
            input_dims=params['INPUT_DIMS'],
            decoders_dist=params['DECODERS_DIST'],
            beta=params['BETA'],
            # alpha=params['ALPHA'],
            use_wasserstein=True,
            dropout_rate=params['DROPOUT_RATE'],
            num_dropouts=params['NUM_DROPOUTS'],
            random_init=params['RANDOM_INIT'],
            text_modalities = text_modalities,
            exhaustive=params['EXHAUSTIVE'],
            deterministic=True
        )
                        
        model  = MACWAE(
            config,            
            encoders=encoders,
            decoders=decoders,
            joint_encoder=joint_encoder
        )
                                          
    elif method == 'JMVAE':
        config = JMVAEConfig(
            n_modalities=params['N_MODALITIES'],
            latent_dim=params['LATENT_DIM'],
            input_dims=params['INPUT_DIMS'],
            decoders_dist=params['DECODERS_DIST'],
            alpha=params['BETA'] # Different name for the reg. param
        )
                
        model  = JMVAE(
            config,
            encoders=encoders,
            decoders=decoders,
            joint_encoder=joint_encoder           
        )
        
    elif method == 'MoPoE':
        config = MoPoEConfig(
            n_modalities=params['N_MODALITIES'],
            latent_dim=params['LATENT_DIM'],
            input_dims=params['INPUT_DIMS'],
            decoders_dist=params['DECODERS_DIST'],
            beta=params['BETA']                
        )
        model = MoPoE(
            config,
            encoders=encoders,
            decoders=decoders             
        )
                
    elif method == 'MVTCAE':
        config = MVTCAEConfig(
            n_modalities=params['N_MODALITIES'],
            latent_dim=params['LATENT_DIM'],
            input_dims=params['INPUT_DIMS'],
            decoders_dist=params['DECODERS_DIST'],
            beta=params['BETA'],
        )
        
        model = MVTCAE(
            config,
            encoders=encoders,
            decoders=decoders             
        )         

    elif method == 'NEXUS':
        
        bottom_dims = {
            mod:params['LATENT_DIM'] for mod in params['INPUT_DIMS'].keys()
        }
        
        config = NexusConfig(
            modalities_specific_dim=bottom_dims,
            n_modalities=params['N_MODALITIES'],
            latent_dim=params['LATENT_DIM'],
            input_dims=params['INPUT_DIMS'],
            decoders_dist=params['DECODERS_DIST'],
            top_beta=params['BETA'],
            bottom_betas={k:1.0 for k in params['INPUT_DIMS'].keys()} # From the Nexus paper - modality specific Betas were set to 1.0
        )
        
        model = Nexus(
            config,
            encoders=encoders,
            decoders=decoders             
        )   
        
    else:
        raise ValueError("Unrecognized method: {}".format(method))
    
    # Set device
    model.to(device)
    
    return config, model