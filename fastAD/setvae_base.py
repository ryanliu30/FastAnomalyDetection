# System imports
import sys

# 3rd party imports
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from abc import abstractmethod

from .anomaly_detection_base import AnomalyDetectionBase

class SetVAEBase(AnomalyDetectionBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        # Initialize soft start parameter
        self.register_buffer("num_iter", torch.tensor(0), persistent=True)

    def training_step(self, batch, batch_idx):
        self.num_iter += 1
        inputs = self.preprocess(batch)
        outputs, mu, log_var = self(inputs)
        loss = self.loss_function(inputs, outputs, mu, log_var)
        self.log("training_loss", loss)
        return loss
    
    def detect(self, inputs):
        z, log_var, *x = self.encode(**inputs)
        outputs = self.decode(z=z, x=x, **inputs)
        return self.reco_loss(inputs, outputs)
        
    def forward(self, inputs):
        mu, log_var, *x = self.encode(**inputs)
        z = self.reparameterize(mu, log_var)
        outputs = self.decode(z=z, x=x, **inputs)
        return outputs, mu, log_var
    
    @abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError("The encode method must be implemented")
    
    @abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError("The decode method must be implemented")
        
    def reparameterize(self, mu, log_var):
        if self.hparams["beta"] > 0:
            return mu + torch.exp(0.5 * log_var) * \
                   torch.randn_like(log_var) * self.warmup
        else:
            return mu
    
    def loss_function(self, inputs, outputs, mu, log_var):
        reco_loss = self.reco_loss(
            inputs,
            outputs
        ).mean()
        kl_div_loss = self.kl_div_loss(mu, log_var).mean()
        self.log_dict({
            "reco_loss": reco_loss,
            "kl_div_loss": kl_div_loss,
        })
        return reco_loss + self.hparams["beta"] * kl_div_loss
    
    def kl_div_loss(self, mu, log_var):
        """
        mu: [N, C]
        log_var: [N, C]
        """        
        loss = - 0.5 * torch.mean(
            1 + log_var - mu ** 2 - torch.exp(log_var),
            dim = -1
        )
        return loss
    
    @property
    def warmup(self):
        return torch.clamp(self.num_iter/self.hparams["soft_kl_steps"], 0, 1)