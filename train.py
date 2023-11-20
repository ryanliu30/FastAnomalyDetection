# Third party import
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from argparse import ArgumentParser
import torch
import yaml

# Local import
from fastAD.models import TransformerSetVAE, TransformerClipVAE, DeepsetSetVAE, DeepsetClipVAE

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--cfg", type = str, required = True)
    parser.add_argument("--accelerator", type = str, default="gpu")
    parser.add_argument("--devices", type = int, default=1)
    parser.add_argument("--train_batches", type = int, default=5000)
    parser.add_argument("--val_batches", type = int, default=1000)
    parser.add_argument("--gradient_clip_val", type = float, default=0.5)
    parser.add_argument("--epochs", type = int, default=100)
    parser.add_argument("--num_sanity_val_steps", type = int, default=0)
    parser.add_argument("--log_period", type = int, default=50)
    parser.add_argument("--log_path", type = str, default="checkpoints")
    parser.add_argument("--project_name", type = str, default="FastAD")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.update(vars(args))

    if cfg["model"] == "TransformerSetVAE":
        model = TransformerSetVAE(cfg)
    elif cfg["model"] == "TransformerClipVAE":
        model = TransformerClipVAE(cfg)
    elif cfg["model"] == "DeepsetSetVAE":
        model = DeepsetSetVAE(cfg)
    elif cfg["model"] == "DeepsetClipVAE":
        model = DeepsetClipVAE(cfg)
    else:
        raise NotImplementedError(f"model {name} specified is not implemented")
        
    torch.compile(model)
    torch.set_float32_matmul_precision('medium')
    
    logger = WandbLogger(
        project = args.project_name, 
        save_dir = args.log_path
    )
    
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        limit_train_batches = args.train_batches,
        limit_val_batches = args.val_batches,
        num_sanity_val_steps = args.num_sanity_val_steps,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[
            ModelCheckpoint(
                monitor='efficiency g-mean',
                mode="max",
                save_top_k=2,
                save_last=True
            )
        ],
        log_every_n_steps = args.log_period,
        default_root_dir = args.log_path
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()