from .. import SetVAEBase
from ..utils import TransformerLayer, get_positional_encoding
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Encoder(nn.Module):
    def __init__(self, 
                 d_kin = 8,
                 d_pid = 8,
                 use_pid = True,
                 d_model = 512,
                 d_ff = 2048,
                 d_output = 128,
                 heads = 8,
                 n_layers = 6,
                 n_pool_layer = 2,
                 dropout = 0,
                 **kwargs,
                ):
        
        super().__init__()
        
        if use_pid:
            d_input = d_kin + d_pid
        else:
            d_input = d_kin
        
        self.ff_input = nn.Sequential(
            nn.Linear(d_input, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
    
        self.ff_mu = nn.Linear(d_model, d_output, bias=True)
        self.ff_var = nn.Linear(d_model, d_output, bias=True)
        
        self.encoder_layers = [
            TransformerLayer(
                d_model = d_model, 
                heads = heads, 
                dropout = dropout,
                d_ff = d_ff
            ) for i in range(n_layers)
        ]
        
        self.pooling_layers = [
            TransformerLayer(
                d_model = d_model, 
                heads = heads, 
                dropout = dropout,
                d_source = d_model,
                d_ff = d_ff,
                src_attn = True,
                self_attn = True
            ) for i in range(n_pool_layer)
        ]
        
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.pooling_layers = nn.ModuleList(self.pooling_layers)
        self.embeddings = Parameter(data = torch.randn((2, 1, d_model)))
        
    def forward(self, x, mask):
        x = self.ff_input(x)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask = ~mask)
        
        z = self.embeddings.expand(-1, x.shape[1], -1)
        for layer in self.pooling_layers:
            z = layer(z, src=x, src_padding_mask = ~mask)
            
        return self.ff_mu(z[0]), self.ff_var(z[1])
    
class Decoder(nn.Module):

    def __init__(self, 
                 d_model = 512,
                 d_ff = 2048,  
                 d_source = 512,
                 d_output = 3,
                 d_pid = 8,
                 use_pid = True,
                 max_length = 128,
                 dropout = 0,
                 heads = 8,
                 n_layers = 8,
                 **kwargs,
                ):
        
        super().__init__()
        
        self.register_buffer(
            'positional_encodings', 
            get_positional_encoding(d_model, max_length), 
            False
        )
        
        self.ff_input = nn.Sequential(
            nn.Linear(d_source, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
        
        if use_pid:
            self.pid_embedding = nn.Embedding(
                d_pid, 
                d_model,
            )
        self.use_pid = use_pid
        
        self.ff_output = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_output, bias=True)
        )
        
        self.decoder_layers = [
            TransformerLayer(
                d_model = d_model, 
                heads = heads, 
                dropout = dropout,
                d_source = d_source,
                d_ff = d_ff,
                src_attn = True
            ) for i in range(n_layers)
        ]
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        

    def forward(self, z, pid, mask):       
        
        x = self.ff_input(z)
        
        if self.use_pid:
            x = x + self.pid_embedding(pid.argmax(-1))
            x = x + self.positional_encodings[
            torch.amax(torch.cumsum(pid, dim = 0) - 1, dim = -1).long()
            ]
        else:
            x = x + self.positional_encodings[:mask.shape[1]].unsqueeze(1)
        
        for layer in self.decoder_layers:
            x = layer(x, padding_mask = ~mask, src=z)
        x = self.ff_output(x)

        return x

class TransformerSetVAE(SetVAEBase):
    
    def __init__(self, hparams):
        super().__init__(hparams)
        encoder_config = hparams.copy()
        encoder_config.update(
            dict(
                n_layers=hparams["n_encoder_layers"], 
                d_output=hparams["d_latent"]
            )
        )
        decoder_config = hparams.copy()
        decoder_config.update(
            dict(
                n_layers=hparams["n_decoder_layers"],
                d_source=hparams["d_latent"]
            )
        )
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)

    def encode(self, mask = None, kin = None, pid = None, **kwargs):
        if self.hparams["use_pid"]:
            x = torch.cat([kin, pid], dim = 2)
        else:
            x = kin.clone()
        z, mu = self.encoder(x.permute(1, 0, 2), mask)
        return z, mu
    
    def decode(self, z = None, pid = None, mask = None, **kwargs):
        z = z.unsqueeze(0)
        pid = pid.permute(1, 0, 2)
        reco_x = self.decoder(z, pid, mask)
        reco_x[..., 0] = reco_x[..., 0].exp()
        return reco_x.permute(1, 0, 2)