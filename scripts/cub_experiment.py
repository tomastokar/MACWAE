import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd

from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.data.utils import set_inputs_to_device

# Insert root path to sys paths
ROOT_PATH = os.path.abspath(os.path.join(__file__ ,"../.."))
sys.path.insert(0, ROOT_PATH)


from utils.auxutils import create_hypermarameters_grid
from utils.evalutils import EvalConfig, Evaluator
from utils.datautils import load_cub, collate_instances
from utils.auxutils import construct_parser, set_device, count_params
from utils.buildutils import build_model
from utils.modelutils import (
    ConvImgEncoder, 
    ConvImgDecoder, 
    TextEncoder,
    TextDecoder,
    SimpleDecoder, 
    SimpleEncoder
)


MAX_WORDS_IN_CAPTIONS = 18
VOCAB_SIZE = 5451
MODALITIES = ['img', 'text', 'label']
N_MODALITIES = len(MODALITIES)  
INPUT_DIMS = {
    'img' : (3, 64, 64),
    'text' : (MAX_WORDS_IN_CAPTIONS, VOCAB_SIZE), 
    'label' : (200,)
}             
DECODERS_DIST = {"img": "normal", "text": "categorical", "label" : "categorical"} 
LOGS_DIR = './logs/'
CAT_HIDDEN_DIM = 32
TEXT_MODALITIES = ['text']
CLASSIFIERS_LATENT_DIM = 128
CLASSIFIERS_TRAINING_EPOCHS = 20
CLASSIFIERS_TRAINING_BATCH_SIZE = 32
NUM_CLASSES = 200

def build_cub_encoders(latent_dim: int):
    
    encoders = {}
    
    encoders['img'] = ConvImgEncoder(
        latent_dim=latent_dim
    )
                    
    encoders['text'] = TextEncoder(                    
        latent_dim = latent_dim,
        max_sentence_length = MAX_WORDS_IN_CAPTIONS,                    
        ntokens = VOCAB_SIZE,
        embed_size=512,
        ff_size=128,
        n_layers=2,
        nhead=2,
        dropout=0.5,
    )  
    
    return encoders

def main(args):
    # Set random seed
    random.seed(args.random_seed)

    # Numpy seed
    np.random.seed(args.random_seed)

    # Torch seed
    torch.manual_seed(args.random_seed)    
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
        
    # Set device
    device = set_device(args.device)
    
    # Make sure output dir exists    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
                
    # Read hyper-parameters
    with open('./configs/hyperparameters.yml', 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)  
        hyperparams = hyperparams['CUB'] 
                    
    # Init data
    data = load_cub()    
                    
    for replicate in range(args.num_replicates):
                                                    
        # Iterate over methods
        for method, params in hyperparams.items():
            
            # Outpit file
            output_file = os.path.join(
                args.output_dir, 
                '{}_rep_{}.csv'.format(method, replicate)
            )
                                    
            # Init results
            results = pd.DataFrame()    
                        
            # Arange list of hyper-parameters candidates
            params_grid = create_hypermarameters_grid(params)
                                                    
            # Iterate through the hyperparameters
            for run, onset in enumerate(params_grid):
                
                try:

                    # Add data related params
                    onset['N_MODALITIES'] = N_MODALITIES
                    onset['INPUT_DIMS'] = INPUT_DIMS
                    onset['DECODERS_DIST'] = DECODERS_DIST
                
                    # ------------
                    # Build model
                    # ------------
                    enc_mod_dim = None
                    dec_mod_dim = None
                    if method in ['MUVAEPlus', 'MUWAEPlus', 'MMVAEPlus']:
                        dec_mod_dim = onset['LATENT_DIM']
                        if method == 'MMVAEPlus':
                            enc_mod_dim = onset['LATENT_DIM']

                    encoders = {}
                    decoders = {}

                    encoders['img'] = ConvImgEncoder(
                        latent_dim=onset['LATENT_DIM'], 
                        modality_specific_dim=enc_mod_dim
                    )
                    decoders['img'] = ConvImgDecoder(
                        latent_dim=onset['LATENT_DIM'], 
                        modality_specific_dim=dec_mod_dim
                    )                      
                                                
                    encoders['text'] = TextEncoder(                    
                        latent_dim = onset['LATENT_DIM'],
                        max_sentence_length = MAX_WORDS_IN_CAPTIONS,                    
                        ntokens = VOCAB_SIZE,
                        modality_specific_dim=enc_mod_dim,
                        embed_size=512,
                        ff_size=128,
                        n_layers=2,
                        nhead=2,
                        dropout=0.5,
                    )
                    
                    decoders['text'] = TextDecoder(                    
                        latent_dim = onset['LATENT_DIM'],
                        modality_specific_dim=dec_mod_dim,
                        input_dim=(MAX_WORDS_IN_CAPTIONS, VOCAB_SIZE)                                        
                    )     
                    
                    encoders['label'] = SimpleEncoder(
                        input_dim=INPUT_DIMS['label'],
                        latent_dim=onset['LATENT_DIM'],
                        hidden_dims=[CAT_HIDDEN_DIM,],
                        modality_specific_dim=enc_mod_dim
                    )
                    decoders['label'] = SimpleDecoder(
                        output_dim=INPUT_DIMS['label'],
                        latent_dim=onset['LATENT_DIM'],
                        hidden_dims=[CAT_HIDDEN_DIM,],
                        modality_specific_dim=dec_mod_dim
                    )                                              
                                            
                    _, model = build_model(
                        method, 
                        params = onset, 
                        encoders = encoders,
                        decoders = decoders,
                        device = device,
                        text_modalities=TEXT_MODALITIES
                    )   
                                        
                    # ------------
                    # Train model
                    # ------------
                    
                    # Trainer config
                    trainer_config = BaseTrainerConfig(
                        num_epochs= onset['EPOCHS'],
                        learning_rate=onset['LEARNING_RATE'],  
                        per_device_train_batch_size=onset['BATCH_SIZE'],
                        per_device_eval_batch_size=onset['BATCH_SIZE'] * 2,
                        output_dir=LOGS_DIR,
                        optimizer_params={'weight_decay' : 1e-5},
                        train_dataloader_num_workers=args.num_workers,
                        eval_dataloader_num_workers=args.num_workers, 
                    )

                    # Init trainer
                    trainer = BaseTrainer(
                        model=model,
                        training_config=trainer_config,
                        train_dataset=data['train'],
                        eval_dataset=data['eval']
                    )
                    
                    # Train model
                    trainer.train()
                    
                    # Take the best model
                    model = trainer._best_model
                    
                    # Keep eval loss
                    eval_loss  = trainer.best_eval_loss
                                            
                    # -------------------------------
                    # Validate model reconstructions
                    # -------------------------------
                    eval_config = EvalConfig(
                        batch_size=256,
                        num_workers=args.num_workers,
                        metrics={'img':'ssim', 'text':'bleu', 'label' : 'acc'},
                        apply_fids=['img'],
                        idxtoword=data['train'].idxtoword
                    )
                                    
                    evaluator = Evaluator(
                        model=model,
                        test_dataset=data['test'],
                        eval_config=eval_config,                  
                    )                

                    eval_results = evaluator.eval()                    

                    # ----------------
                    # Collect results 
                    # ----------------
                    result = pd.Series(onset)
                    result['RUN'] = run
                    result['EVAL_LOSS'] = eval_loss
                    result['NUM_PARAMS'] = count_params(model)
                    
                    # Add reconstruction errors
                    for m, v in eval_results['reconstruction'].items():
                        result['RECONSTRUCTION_{}'.format(m)] = v
                        
                    # Add translation errors
                    for m, v in eval_results['m2o_translation'].items():
                        result['MANY_TO_1_TRANSLATION_{}'.format(m)] = v
                    
                    # Add translation errors
                    for m1, r in eval_results['o2m_translation'].items():
                        for m2, v in r.items():
                            result['1_TO_M_TRANSLATION_{}_{}'.format(m1, m2)] = v               
                    
                    # Remove non-keepers   
                    del(result['N_MODALITIES']) 
                    del(result['INPUT_DIMS'])
                    del(result['DECODERS_DIST'])
                    
                    # Show results
                    print(result)
                    
                    # Add to results
                    results = pd.concat(
                        [results, result.to_frame().T], 
                        ignore_index = True
                    )
                    
                    # Save to file                
                    results.to_csv(output_file)
                    
                except:
                    
                    continue

        
if __name__ == '__main__':    
    parser = construct_parser()
    args = parser.parse_args()
    main(args)