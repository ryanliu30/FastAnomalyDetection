from .. import SetVAEBase
from ..utils import DeepSetLayer, get_positional_encoding
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
                 n_layers = 6,
                 dropout = 0,
                 **kwargs,
                ):
        
        super().__init__()
        
        if use_pid:
            d_input = d_kin + d_pid
        else:
            d_input = d_kin
        
        self.ff_input = nn.Linear(d_input, d_model, bias=True)
        self.ff_mu = nn.Linear(d_model, d_output, bias=True)
        self.ff_var = nn.Linear(d_model, d_output, bias=True)
        
        self.encoder_layers = [
            DeepSetLayer(
                d_model = d_model, 
                dropout = dropout,
                d_ff = d_ff
            ) for i in range(n_layers)
        ]
        
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        
    def forward(self, x, mask):
        x = self.ff_input(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        weights = (mask.float() / mask.sum(1, keepdim = True))[:, :, None]
        z = (x * weights).sum(1)
        return self.ff_mu(z), self.ff_var(z)
    
class Decoder(nn.Module):

    def __init__(self, 
                 d_model = 512,
                 d_ff = 2048,  
                 d_source = 512,
                 d_output = 3,
                 use_pid = True,
                 d_pid = 8,
                 max_length = 128,
                 dropout = 0,
                 n_layers = 8,
                 **kwargs,
                ):
        
        super().__init__()
        
        self.register_buffer(
            'positional_encodings', 
            get_positional_encoding(d_model, max_length),
            False
        )
        
        self.ff_input = nn.Linear(d_source, d_model, bias=True)
        self.ff_output = nn.Linear(d_model, d_output, bias=True)
        
        if use_pid:
            self.pid_embedding = nn.Embedding(
                d_pid, 
                d_model,
            )
        self.use_pid = use_pid
        
        self.decoder_layers = [
            DeepSetLayer(
                d_model = d_model, 
                dropout = dropout,
                d_ff = d_ff
            ) for i in range(n_layers)
        ]
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        

    def forward(self, z, pid, mask):  
        
        x = self.ff_input(z)
        
        if self.use_pid:
            x = x + self.pid_embedding(pid.argmax(-1))
            x = x + self.positional_encodings[
            torch.amax(torch.cumsum(pid, dim = 1) - 1, dim = -1).long()
            ]
        else:
            x = x + self.positional_encodings[:mask.shape[1]].unsqueeze(0)
        
        for layer in self.decoder_layers:
            x = layer(x, mask)
            
        x = self.ff_output(x)
        return x

class DeepsetSetVAE(SetVAEBase):
    
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
        z, mu = self.encoder(x, mask)
        return z, mu
    
    def decode(self, z = None, pid = None, mask = None, **kwargs):
        z = z.unsqueeze(1)
        reco_x = self.decoder(z, pid, mask)
        reco_x[..., 0] = reco_x[..., 0].exp()
        return reco_x
        
        