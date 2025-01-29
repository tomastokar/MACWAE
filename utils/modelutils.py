import time
import math
import torch
import numpy as np
import torch.nn as nn

from multivae.models.base import (
    BaseEncoder, 
    BaseDecoder, 
    ModelOutput
)

class Unflatten(nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)


class SimpleImageEncoder(BaseEncoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim: int, modality_specific_dim: int = None, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.modality_specific_dim = modality_specific_dim
        
        if self.modality_specific_dim is not None:
            total_dim = self.latent_dim + self.modality_specific_dim
            
            # Private branch
            self.style_mu = nn.Linear(total_dim, self.modality_specific_dim)
            self.style_logvar = nn.Linear(total_dim, self.modality_specific_dim)
            
        else:
            total_dim = self.latent_dim

        # Shared branch
        self.mu = nn.Linear(total_dim, self.latent_dim)
        self.logvar = nn.Linear(total_dim, self.latent_dim)

        # Shared encoder        
        self.encoder = nn.Sequential(                                         # input shape (input_channels, 28, 28)
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),             # -> (32, 14, 14)
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),            # -> (64, 7, 7)
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),           # -> (128, 4, 4)
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(start_dim=1),                                    # -> (2048)
            nn.Linear(2048, total_dim),    
            nn.ReLU(),
        )
        

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.encoder(x)
        if self.modality_specific_dim is not None:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h), 
                style_embedding = self.style_logvar(h), 
                style_log_covariance = self.style_mu(h)
            )
        else:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h)
            )
        return output


class SimpleImageDecoder(BaseDecoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim: int, modality_specific_dim: int = None, output_channels: int = 3):
        super().__init__()
        self.output_channels = output_channels
        self.output_dim = (self.output_channels, 28, 28)
        self.latent_dim = latent_dim
        self.modality_specific_dim = modality_specific_dim

        if self.modality_specific_dim is not None:
            self.input_dim = self.latent_dim + self.modality_specific_dim
            
        else:
            self.input_dim = self.latent_dim


        self.decoder = nn.Sequential( 
            nn.Linear(self.input_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 4)),                                      # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (output_channels, 28, 28)
        )

    def forward(self, z: torch.Tensor):                
        output_shape = (*z.shape[:-1],) + self.output_dim
        h = z.reshape(-1, self.input_dim)        
        x_hat = self.decoder(h)            
        x_hat = torch.sigmoid(x_hat)            
        x_hat = x_hat.reshape(output_shape)
        return ModelOutput(reconstruction = x_hat)


class ConvImgEncoder(BaseEncoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim: int, modality_specific_dim: int = None):
        super().__init__()

        self.latent_dim = latent_dim
        self.modality_specific_dim = modality_specific_dim
        
        if self.modality_specific_dim is not None:
            total_dim = self.latent_dim + self.modality_specific_dim
            
            # Private branch
            self.style_mu = nn.Linear(total_dim, self.modality_specific_dim)
            self.style_logvar = nn.Linear(total_dim, self.modality_specific_dim)
            
        else:
            total_dim = self.latent_dim

        # Shared branch
        self.mu = nn.Linear(total_dim, self.latent_dim)
        self.logvar = nn.Linear(total_dim, self.latent_dim)

        # Shared encoder        
        self.encoder = nn.Sequential(                                         # input shape (3, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),             # -> (32, 32, 32)
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),            # -> (64, 16, 16)
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),           # -> (128, 8, 8)
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),          # -> (256, 4, 4)
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(start_dim=1),                                    # -> (4096 = 256 * 4 * 4)
            nn.Linear(4096, total_dim),    
            nn.ReLU(),
        )
        

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.encoder(x)
        if self.modality_specific_dim is not None:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h), 
                style_embedding = self.style_logvar(h), 
                style_log_covariance = self.style_mu(h)
            )
        else:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h)
            )
        return output


class ConvImgDecoder(BaseDecoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim: int, modality_specific_dim: int = None):
        super().__init__()
        self.output_dim = (3, 64, 64)
        self.latent_dim = latent_dim
        self.modality_specific_dim = modality_specific_dim

        if self.modality_specific_dim is not None:
            self.input_dim = self.latent_dim + self.modality_specific_dim
            
        else:
            self.input_dim = self.latent_dim


        self.decoder = nn.Sequential( 
            nn.Linear(self.input_dim, 4096),                                                   # -> (4096)
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),                                 # -> (256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),# -> (64, 7, 7)
            # nn.BatchNorm2d(128),
            nn.ReLU(),            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (64, 7, 7)
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, z: torch.Tensor):   
        output_shape = (*z.shape[:-1],) + self.output_dim
        h = z.reshape(-1, self.input_dim)        
        x_hat = self.decoder(h)            
        x_hat = torch.sigmoid(x_hat)            
        x_hat = x_hat.reshape(output_shape)
        return ModelOutput(reconstruction = x_hat)


class SimpleEncoder(BaseEncoder):    
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int,
                 hidden_dims: list,  
                 modality_specific_dim: int = None):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.modality_specific_dim = modality_specific_dim
        
        # Shared encoder        
        self.encoder = []
        input_dim = int(np.prod(self.input_dim))
        for dim in self.hidden_dims:
            self.encoder.append(nn.Linear(input_dim, dim))
            # self.encoder.append(nn.BatchNorm1d(dim))
            self.encoder.append(nn.ReLU())
            input_dim = dim
            
        # Make sequential
        self.encoder = nn.Sequential(*self.encoder)
                
        # Shared branch
        self.mu = nn.Linear(
            self.hidden_dims[-1], 
            self.latent_dim
        )
        
        self.logvar = nn.Linear(
            self.hidden_dims[-1], 
            self.latent_dim
        )
        
        # Private branch
        if self.modality_specific_dim is not None:            
            
            self.style_mu = nn.Linear(
                self.hidden_dims[-1], 
                self.modality_specific_dim
            )
            
            self.style_logvar = nn.Linear(
                self.hidden_dims[-1], 
                self.modality_specific_dim
            )            
        

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.encoder(x)
        if self.modality_specific_dim is not None:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h), 
                style_embedding = self.style_mu(h), 
                style_log_covariance = self.style_logvar(h)
            )
        else:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h)
            )
        return output


class SimpleDecoder(BaseDecoder):    
    def __init__(self, 
                 output_dim: int,  
                 latent_dim: int,
                 hidden_dims: [],
                 modality_specific_dim: int = None):
        super().__init__()

        self.output_dim = output_dim        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.modality_specific_dim = modality_specific_dim
        
        self.total_latent_dim = self.latent_dim
        if self.modality_specific_dim is not None:
            self.total_latent_dim += self.modality_specific_dim
        
        # Decoder     
        self.decoder = []
        input_dim = self.total_latent_dim
        for dim in self.hidden_dims:
            self.decoder.append(nn.Linear(input_dim, dim))
            # self.decoder.append(nn.BatchNorm1d(dim))
            self.decoder.append(nn.ReLU())
            input_dim = dim

        # Make sequential
        self.decoder = nn.Sequential(*self.decoder)

        # Add final layer            
        self.decoder.append(nn.Linear(input_dim, int(np.prod(self.output_dim))))
        self.decoder.append(nn.Sigmoid())
                
    def forward(self, z: torch.Tensor) -> ModelOutput:  
        output_shape = (*z.shape[:-1],) + self.output_dim         
        h = z.reshape(-1, self.total_latent_dim)
        x_hat = self.decoder(h)    
        x_hat = x_hat.reshape(output_shape)
        return ModelOutput(reconstruction = x_hat)


class PositionalEncoding(nn.Module):
    """Taken from torch/examples"""

    def __init__(self, embedding_dim: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        # self.pe = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(1, max_len, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TextEncoder(BaseEncoder):
    def __init__(
        self,
        latent_dim: int, 
        max_sentence_length: int,
        ntokens: int,
        modality_specific_dim: int = None,
        embed_size: int = 512,
        nhead: int = 4,
        ff_size: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.5,
    ):
        """
        A transformer-based encoder for text.

        Parameters:
            ntokens (int): Vocabulary size.
            embed_size (int): Size of the token embedding vectors. Default: 512
            nhead (int): Number of head in the MultiHeadAttention module. Default: 8
            ff_size (int): Number of units in the feedforward layers. Default: 1024
            n_layers (int): Number of Encoders layers in the TransformerEncoder. Default: 8
            dropout (float): Dropout rate. Default: 0.5
        """

        BaseEncoder.__init__(self)
        self.modality_specific_dim = modality_specific_dim
        self.latent_dim = latent_dim
        self.embed_size = embed_size

        self.token_embedding = nn.Embedding(ntokens, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_sentence_length, dropout)        
        encoder_layer = nn.TransformerEncoderLayer(
            embed_size, nhead, ff_size, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Shared branch
        self.mu = nn.Linear(embed_size * max_sentence_length, self.latent_dim)
        self.log_covariance = nn.Linear(
            embed_size * max_sentence_length, self.latent_dim
        )
                
        # Private branch
        if self.modality_specific_dim is not None:            
            self.style_mu = nn.Linear(embed_size * max_sentence_length, self.modality_specific_dim)
            self.style_log_covariance = nn.Linear(
                embed_size * max_sentence_length, self.modality_specific_dim
            )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.token_embedding.weight, -initrange, initrange)

    def forward(self, inputs):
        src = inputs["tokens"]        
        padding_mask = inputs["padding_mask"]

        src = self.token_embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        
        transformer_output = self.transformer_encoder(
            src, 
            src_key_padding_mask=~padding_mask.bool()
        )
       
       
        h = transformer_output.reshape(src.shape[0], -1)
        
        if self.modality_specific_dim is not None:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.log_covariance(h), 
                style_embedding = self.style_mu(h), 
                style_log_covariance = self.style_log_covariance(h),
                transformer_output=transformer_output
            )
        else:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.log_covariance(h),
                transformer_output=transformer_output
            )
       
        return output


class TextDecoder(BaseDecoder):
    def __init__(
            self,
            latent_dim: int,
            input_dim: tuple,
            modality_specific_dim: int = None            
        ):
        """
        A simple MLP decoder for text.
        """
        BaseDecoder.__init__(self)

        self.modality_specific_dim = modality_specific_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
                
        self.total_latent_dim = self.latent_dim
        if self.modality_specific_dim is not None:
            self.total_latent_dim += self.modality_specific_dim
        
        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(self.total_latent_dim, 512), nn.ReLU()))
        layers.append(nn.Sequential(nn.Linear(512, int(np.prod(input_dim))))) # Shouldn't there be a sigmoid?

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor):        
        output = ModelOutput()

        max_depth = self.depth
        out = z        
        for i in range(max_depth):
            out = self.layers[i](out)
            if i + 1 == self.depth:
                # output_shape = (*z.shape[:-1],) + self.output_dim
                output_shape = (*z.shape[:-1],) + self.input_dim
                output["reconstruction"] = out.reshape(output_shape)
        return output


class SimpleSoundEncoder(BaseEncoder):

    def __init__(self, latent_dim: int, modality_specific_dim: int = None, input_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.modality_specific_dim = modality_specific_dim
        
        if self.modality_specific_dim is not None:
            total_dim = self.latent_dim + self.modality_specific_dim
            
            # Private branch
            self.style_mu = nn.Linear(total_dim, self.modality_specific_dim)
            self.style_logvar = nn.Linear(total_dim, self.modality_specific_dim)
            
        else:
            total_dim = self.latent_dim

        # Shared branch
        self.mu = nn.Linear(total_dim, self.latent_dim)
        self.logvar = nn.Linear(total_dim, self.latent_dim)

        # Shared encoder        
        self.encoder = nn.Sequential(                                         # input shape (input_channels, 28, 28)
            nn.Conv2d(self.input_channels, 32, kernel_size=(1, 128), stride=1, padding=0),             # -> (32, 32, 1)
            # nn.BatchNorm2d()
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),            # -> (64, 16, 1)
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),           # -> (128, 8, 1)
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(start_dim=1),                                    # -> (1024)
            nn.Linear(1024, total_dim),    
            nn.ReLU(),
        )
        

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.encoder(x)
        if self.modality_specific_dim is not None:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h), 
                style_embedding = self.style_logvar(h), 
                style_log_covariance = self.style_mu(h)
            )
        else:
            output = ModelOutput(
                embedding = self.mu(h),
                log_covariance = self.logvar(h)
            )
        return output


class SimpleSoundDecoder(BaseDecoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, latent_dim: int, modality_specific_dim: int = None, output_channels: int = 1):
        super().__init__()
        self.output_channels = output_channels
        self.output_dim = (self.output_channels, 32, 128)
        self.latent_dim = latent_dim
        self.modality_specific_dim = modality_specific_dim

        if self.modality_specific_dim is not None:
            self.input_dim = self.latent_dim + self.modality_specific_dim
            
        else:
            self.input_dim = self.latent_dim


        self.decoder = nn.Sequential( 
            nn.Linear(self.input_dim, 1024),                                # -> (1920)
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(128, 8, 1)),                                      # -> (128, 8, 1)
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),                   # -> (64, 16, 1)
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),  # -> (32, 32, 1)
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_channels, kernel_size=(1, 128), stride=1, padding=0),   # -> (output_channels, 32, 128)
        )

    def forward(self, z: torch.Tensor):                
        output_shape = (*z.shape[:-1],) + self.output_dim
        h = z.reshape(-1, self.input_dim)        
        x_hat = self.decoder(h)            
        x_hat = torch.sigmoid(x_hat)            
        x_hat = x_hat.reshape(output_shape)
        return ModelOutput(reconstruction = x_hat)