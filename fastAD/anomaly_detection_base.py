# System imports
import sys

# 3rd party imports
from lightning.pytorch.core import LightningModule
import numpy as np
import pandas as pd
import torch
from weaver.utils.dataset import SimpleIterDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import yaml
import wandb
import glob
from abc import ABC
from abc import abstractmethod
import plotly.express as px

def load_files(flist):
    file_dict = {}
    for f in flist:
        if ':' in f:
            name, fp = f.split(':')
        else:
            name, fp = '_', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)
    return file_dict

class AnomalyDetectionBase(ABC, LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module
        """
        self.save_hyperparameters(hparams)
        self.data_config_file = "fastAD/configs/dataset_config.yaml"
        self.loader_config_file = 'fastAD/configs/loader_config.yaml'
        
        # Initialize ouptuts
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def train_dataloader(self):
        with open(self.loader_config_file) as f:
            flist = yaml.load(f, Loader=yaml.FullLoader)["data_train"]
        file_dict = load_files(flist)
        dataset = SimpleIterDataset(
            file_dict, 
            self.data_config_file,
            for_training=True,
            remake_weights=True,
            fetch_step=0.01,
            infinity_mode=False,
            name='train'
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            drop_last=True,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True
        )

    def val_dataloader(self):
        with open(self.loader_config_file) as f:
            flist = yaml.load(f, Loader=yaml.FullLoader)["data_val"]
        file_dict = load_files(flist)
        dataset = SimpleIterDataset(
            file_dict, 
            self.data_config_file,
            for_training=True,
            remake_weights=True,
            fetch_step=0.01,
            infinity_mode=False,
            name='validation'
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            drop_last=True,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True
        )

    def test_dataloader(self):
        with open(self.loader_config_file) as f:
            flist = yaml.load(f, Loader=yaml.FullLoader)["data_test"]
        file_dict = load_files(flist)
        dataset = SimpleIterDataset(
            file_dict, 
            self.data_config_file,
            for_training=False,
            remake_weights=True,
            fetch_step=0.01,
            infinity_mode=False,
            name='test'
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            drop_last=False,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True
        )
    
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"]
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    
    def preprocess(self, batch):
        pmu, kin, pid, dsp, mask = (
            batch[0]["pmu"],
            batch[0]["kin"],
            batch[0]["pid"],
            batch[0]["dsp"],
            batch[0]["mask"]
        )
        mask = mask.squeeze(1).contiguous().bool()
        max_len = mask.sum(-1).max()
        pmu = pmu.permute(0, 2, 1).contiguous()[:, :max_len] 
        kin = kin.permute(0, 2, 1).contiguous()[:, :max_len]
        pid = pid.permute(0, 2, 1).contiguous()[:, :max_len]
        dsp = dsp.permute(0, 2, 1).contiguous()[:, :max_len]
        mask = mask[:, :max_len]
        return {
            "pmu": pmu,
            "kin": kin,
            "pid": pid,
            "dsp": dsp,
            "mask": mask
        }

    def l2_loss(self, x, y):
        """
        Compute L2 loss in spherical coordinates, which is defined as 
        |pT' - pT|^2 - pT^2 * [(eta - eta')^2 + (phi - phi')^2] / R_0^2
        """
        return (
            (x[..., 0] - y[..., 0]).square() + 
            x[..., 0].square() * ((x[..., 1:3] - y[..., 1:3]) / 
            self.hparams["angular_scale"]).square().sum(-1)
        )
    
    def reco_loss(self, inputs, y):

        pid = inputs["pid"].argmax(-1)
        mask = inputs["mask"]
        x = inputs["kin"].clone()
        weights = mask.float() / mask.float().sum(1, keepdim = True)
        y = y[:, :x.shape[1]]
        
        x.masked_fill_(~mask[:, :, None], 0)
        y.masked_fill_(~mask[:, :, None], 0)
        dist_mat = self.l2_loss(x[:, :, None], y[:, None, :])
        if self.hparams["use_pid"]:
            dist_mat.masked_fill_(
                ~(mask[:, :, None] & mask[:, None, :]) | \
                (pid[:, :, None] != pid[:, None, :]),
                dist_mat.max().item() + 1
            )
        else:
            dist_mat.masked_fill_(
                ~(mask[:, :, None] & mask[:, None, :]),
                dist_mat.max().item() + 1
            )
        loss = 0.5 * weights * (
            torch.amin(dist_mat, dim = 1) + 
            torch.amin(dist_mat, dim = 2)
        )
        
        return loss.sum(-1)
    
    @abstractmethod
    def detect(self, inputs):
        raise NotImplemetedError("implement anomaly detection method!")
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplemetedError("implement training loop!")
    
    def shared_evaluation(self, batch, batch_idx, log=False):
        inputs = self.preprocess(batch)
        labels = batch[1]["_label_"]
        tau21 = batch[2]["jet_tau2"]/(1e-8 + batch[2]["jet_tau1"])
        tau32 = batch[2]["jet_tau3"]/(1e-8 + batch[2]["jet_tau2"])
        tau42 = batch[2]["jet_tau4"]/(1e-8 + batch[2]["jet_tau2"])
        tau = torch.sigmoid(9.9 - 7.2 * tau21 - 3.3 * tau32 - 3.4 * tau42)

        scores = self.detect(inputs)
      
        return (
            labels.cpu().numpy(),
            tau.cpu().numpy(),
            scores.cpu().numpy(),
        )

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(
            self.shared_evaluation(batch, batch_idx, log=True)
        )

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(
            self.shared_evaluation(batch, batch_idx, log=True)
        )
        
    def log_AD_metrics(self, labels, scores, tau):        

        label_names = [
            "H->bb",
            "H->cc",
            "H->gg",
            "H->4q",
            "H->qql",
            "Z->qq",
            "W->qq",
            "t->bqq",
            "t->bl"
        ]
        df = pd.DataFrame()
        effs = []
        tau_effs = []
        for i, label in zip(range(1, 10), label_names):
            mask = (labels == i) | (labels == 0)
            
            fpr, tpr, _ = roc_curve((labels[mask] == i), scores[mask])
            eff = np.interp([0.01], fpr, tpr).item()
            effs.append(eff)
            tau_fpr, tau_tpr, _ = roc_curve((labels[mask] == i), tau[mask])
            tau_eff = np.interp([0.01], tau_fpr, tau_tpr).item()
            tau_effs.append(tau_eff)
            
            fpr, tpr = fpr[(tpr > 0) & (fpr > 0)], tpr[(tpr > 0) & (fpr > 0)]
            tau_fpr, tau_tpr = (
                tau_fpr[(tau_tpr > 0) & (tau_fpr > 0)],
                tau_tpr[(tau_tpr > 0) & (tau_fpr > 0)]
            )
            df = pd.concat(
                [
                    df, 
                    pd.DataFrame({"background rejection": 1/fpr,
                    "signal efficiency": tpr,
                    "relative efficiency": tpr/np.interp(fpr, tau_fpr, tau_tpr),
                    "jet type": label})
                ]
            , ignore_index=True)
            
            auc = roc_auc_score((labels[mask] == i), scores[mask])
            self.log(f"{label}: eff", eff)
            self.log(f"{label}: rel_eff", eff/(tau_eff+1e-8))
            self.log(f"{label}: auc", auc)
            
        effs, tau_effs = np.array(effs), np.array(tau_effs)
        self.log(
            "efficiency g-mean",
            np.exp(np.mean(np.log(effs)))
        )
        self.log(
            "relative efficiency g-mean",
            np.exp(np.mean(np.log(effs)) - np.mean(np.log(tau_effs)))
        )
        
        color_seq = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf'
        ]
        fig1 = px.line(
            pd.concat(
                [
                    df, 
                    pd.DataFrame({
                        "background rejection": 1 / np.linspace(1e-3, 1, 1000),
                        "signal efficiency": np.linspace(1e-3, 1, 1000),
                        "jet type": "chance level"
                    })
                ]
            , ignore_index=True),
            x="signal efficiency",
            y="background rejection",
            line_group='jet type',
            color='jet type',
            color_discrete_sequence=color_seq,
            log_y = True
        )
        fig2 = px.line(
            df,
            x="background rejection",
            y="relative efficiency",
            line_group='jet type',
            color='jet type',
            color_discrete_sequence=color_seq,
            log_x = True,
            log_y = True
        )
        
        wandb.log({
            "ROC Curves": fig1,
            "Relative Efficiency": fig2
        })
    
    def shared_on_epoch_end_eval(self, preds):
        # Transpose list of lists
        labels, tau, scores = map(
            lambda x: np.concatenate(list(x), axis = 0), zip(*preds)
        )
        
        # Evaluate model performance
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        auc = roc_auc_score((labels != 0), scores)
        self.log_dict({
            f"auc": auc,
        })
        self.log_AD_metrics(labels, scores, tau)
        
    def on_validation_epoch_end(self):
        self.shared_on_epoch_end_eval(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        self.shared_on_epoch_end_eval(self.test_step_outputs)
        self.test_step_outputs.clear()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become 
        built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()