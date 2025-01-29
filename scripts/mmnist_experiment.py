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
from utils.modelutils import SimpleImageEncoder, SimpleImageDecoder, MultiClassifierTrainer
from utils.evalutils import EvalConfig, Evaluator
from utils.datautils import load_polymnist
from utils.auxutils import construct_parser, set_device, count_params
from utils.buildutils import build_model

MODALITIES = ['m0', 'm1', 'm2', 'm3', 'm4']
N_MODALITIES = len(MODALITIES)     
INPUT_DIMS = {m : (3,28,28) for m in MODALITIES}             
DECODERS_DIST = {m : 'normal' for m in MODALITIES}               
LOGS_DIR = './logs/'
CLASSIFIERS_LATENT_DIM = 32
CLASSIFIERS_TRAINING_EPOCHS = 10
CLASSIFIERS_TRAINING_BATCH_SIZE = 32
NUM_CLASSES = 10

def main(args):
    # Set random seed
    random.seed(args.random_seed)

    # Numpy seed
    np.random.seed(args.random_seed)

    # Torch seed
    torch.manual_seed(args.random_seed)    
    torch.cuda.manual_seed(args.random_seed)
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
        hyperparams = hyperparams['MMNIST'] 
                    
    # Init data
    data = load_polymnist()
                    
    for replicate in range(args.num_replicates):        
                                                
        # Iterate of methods
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
                for m in MODALITIES:
                    encoders[m] = SimpleImageEncoder(
                        onset['LATENT_DIM'], 
                        modality_specific_dim=enc_mod_dim
                    )
                    decoders[m] = SimpleImageDecoder(
                        onset['LATENT_DIM'], 
                        modality_specific_dim=dec_mod_dim
                    )                
                
                _, model = build_model(
                    method, 
                    params = onset, 
                    encoders = encoders,
                    decoders = decoders,
                    device = device
                )   
                                    
                # ------------
                # Train model
                # ------------
                
                # Trainer config
                trainer_config = BaseTrainerConfig(
                    num_epochs= onset['EPOCHS'],
                    learning_rate=onset['LEARNING_RATE'],  
                    per_device_train_batch_size=onset['BATCH_SIZE'],
                    per_device_eval_batch_size=onset['BATCH_SIZE'],
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
                    batch_size=256, # Was 1024
                    num_workers=args.num_workers,
                    metrics={m:'ssim' for m in MODALITIES},
                    apply_fids=MODALITIES                  
                )
                                
                evaluator = Evaluator(
                    model=model,
                    test_dataset=data['test'],
                    eval_config=eval_config
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
                        result['1_TO_MANY_TRANSLATION_{}_{}'.format(m1, m2)] = v                   

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


        
if __name__ == '__main__':    
    parser = construct_parser()
    args = parser.parse_args()
    main(args)