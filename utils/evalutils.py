import time
import torch
import torch.distributions as dist
from dataclasses import field
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader
from pythae.models.base.base_utils import ModelOutput
from multivae.data.datasets import MultimodalBaseDataset
from multivae.models.base import BaseMultiVAE
from multivae.data.utils import set_inputs_to_device
from torcheval.metrics import StructuralSimilarity as SSIM
from torcheval.metrics import MulticlassAccuracy, MeanSquaredError, BLEUScore
from torch.nn.functional import cross_entropy


@dataclass
class EvalConfig:
    batch_size: int = 128
    num_workers: int = 0
    metrics: dict = None
    idxtoword: dict = None
    padding_token: int = 1
    eps: float = 0.01
    apply_fids: list = field(default_factory=lambda: [])
    
def gaussian_nll(input, target):
    return -dist.Normal(input, 1.0).log_prob(target).mean()


def categorical_nll(input, target):
    return cross_entropy(input, target)


def BLEU4Score(device = None):
    return BLEUScore(n_gram=4, device = device)


class Evaluator:
    def __init__(self,
                 model : BaseMultiVAE,
                 test_dataset: MultimodalBaseDataset,
                 eval_config: EvalConfig,
                 eps = 0.01):
        
        self.model = model                
        self.device = self.model.device        
        self.test_dataset = test_dataset
        self.eval_config = eval_config
        
        self.modalities = list(self.model.decoders.keys())
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.eval_config.batch_size,
            num_workers=self.eval_config.num_workers
        )
        
        self.num_batches = len(self.test_loader)
        
        self.metric_funcs = {
            'acc' : MulticlassAccuracy,
            'mse' : MeanSquaredError,
            'ssim' : SSIM,
            'bleu' : BLEU4Score
        }
        
        self.metrics = {
            mod:fn for mod, fn in self.eval_config.metrics.items() 
            if fn in self.metric_funcs.keys()
        }
                                            
    def report(self, batch):
        elapsed_time = time.time() - self.start_time
        print('Evaluating batch {}/{}; Elapsed time: {:1.2f} seconds!'.format(batch+1, self.num_batches, elapsed_time))
    
    def token2str(self, batch_tokens):
        
        batch_tokens = batch_tokens.tolist()
        
        strings = []
        for tokens in batch_tokens:
            words = [self.eval_config.idxtoword[t] for t in tokens if t not in [0, 1]]
            strings.append(' '.join(words))
            
        return strings
        
    def process_text_preds(self, preds):
        tokens = preds.argmax(dim = -1)            
        padding = torch.ones_like(tokens)
        padding[tokens == self.eval_config.padding_token] = 0.
        return {'tokens' : tokens, 'padding_mask' : padding}
    
    def reconstruction_from_subset(self, gen_mod: list, cond_mod: list):
                    
        metrics = {
            m : self.metric_funcs[self.metrics[m]](device = self.device) for m in gen_mod
        }        
        
        print('\n')
        self.start_time = time.time()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                            
                # Set batch to device
                batch = set_inputs_to_device(batch, self.device)
                
                # Make prediction
                output = self.model.predict(
                    batch, 
                    cond_mod = cond_mod, 
                    gen_mod = gen_mod
                )
                
                for mod in gen_mod:
                    preds = output[mod]
                    target = batch.data[mod]
                    
                    if isinstance(metrics[mod], MulticlassAccuracy):
                        
                        # Prep target
                        target = target.argmax(dim=1)
                        
                        
                    if isinstance(metrics[mod], MeanSquaredError):
                        
                        if len(target.shape) > 3:
                        
                            # Prep target
                            target = target.flatten(start_dim=1)      
                        
                        if len(preds.shape) > 3:
                            
                            # Prep predictions
                            preds = preds.flatten(start_dim=1)      
                            

                    elif isinstance(metrics[mod], BLEUScore):
                        
                        # Prep predictions
                        preds = self.process_text_preds(preds)
                        preds = self.token2str(preds['tokens']) 
                        
                        # Prep targets
                        target = target['tokens']                              
                        target = self.token2str(target)
                                                                    
                    metrics[mod].update(preds, target)
                    
                # Report progress  
                self.report(i)
            
            results = {
                m : metrics[m].compute().item() for m in gen_mod
            }
        
        return results    
                    
    def eval_reconstructions(self):
        # Joint reconstruction with all modalities
        print('\n == Evaluating reconstructions ==')
        return self.reconstruction_from_subset(
            gen_mod=self.modalities,
            cond_mod=self.modalities
        )
                
    def eval_translation(self):

        # Continers        
        m2o_translation = {} # one-to-many
        o2m_translation = {} # many-to-one

        # Iterate over modelities
        for one_mod in self.modalities:
            print('\n == Testing many-to-1 translation to {} =='.format(one_mod))
            many_mod = [m for m in self.modalities if m != one_mod]
            m2o_translation[one_mod] = self.reconstruction_from_subset(
                gen_mod=[one_mod],
                cond_mod=many_mod
            )[one_mod]
            
            
            print('\n == Testing 1-to-many translation from {} =='.format(one_mod))
            recons = self.reconstruction_from_subset(
                gen_mod=many_mod,
                cond_mod=[one_mod]
            )            
            
            o2m_translation[one_mod] = {
                mod:recons[mod] for mod in many_mod
            }

        return m2o_translation, o2m_translation
        
                                    
    def eval(self):
        
        # Set model to eval mode
        self.model.eval()
        
        # Evaluate reconstructions
        recons = self.eval_reconstructions()
        
        # Evaluate modality translation
        m2o_trans, o2m_trans = self.eval_translation()
                                            
        return ModelOutput(
            reconstruction = recons, 
            m2o_translation = m2o_trans,
            o2m_translation = o2m_trans
        )            