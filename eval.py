# Third party import
from argparse import ArgumentParser
import yaml
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from itertools import islice

# Local import
from fastAD.models import TransformerSetVAE, TransformerClipVAE, DeepsetSetVAE, DeepsetClipVAE

matplotlib.use("agg")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--ckpt", type = str, required = True)
    parser.add_argument("--save_name", type = str, required = True)
    args = parser.parse_args()
    return args

def tocuda(x):
    if isinstance(x, list):
        return [tocuda(y) for y in x]
    if isinstance(x, dict):
        return {name: tocuda(x[name]) for name in x}
    else:
        return x.cuda()

@torch.no_grad()
def get_predictions(model):
    model.eval()
    dfs = []
    loader = model.test_dataloader()
    for batch in tqdm(loader, total = 20000000 // model.hparams["batch_size"] + 1):
        inputs = model.preprocess(tocuda(batch))
        tau21 = batch[2]["jet_tau2"]/(1e-8 + batch[2]["jet_tau1"])
        tau32 = batch[2]["jet_tau3"]/(1e-8 + batch[2]["jet_tau2"])
        tau42 = batch[2]["jet_tau4"]/(1e-8 + batch[2]["jet_tau2"])
        batch[2]["tau_scores"] = 9.9 - 7.2 * tau21 - 3.3 * tau32 - 3.4 * tau42
        batch[2]["is_signal"] = batch[1]["_label_"] > 0
        df = pd.DataFrame({
            "scores": model.detect(inputs).cpu().numpy(),
            "labels": batch[1]["_label_"],
            **batch[2]
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index = True)
    
def main():
    args = parse_arguments()

    CMS = {
        "font.family": "sans-serif",
        "mathtext.fontset": "custom",
        "mathtext.rm": "TeX Gyre Heros",
        "mathtext.bf": "TeX Gyre Heros:bold",
        "mathtext.sf": "TeX Gyre Heros",
        "mathtext.it": "TeX Gyre Heros:italic",
        "mathtext.tt": "TeX Gyre Heros",
        "mathtext.cal": "TeX Gyre Heros",
        "mathtext.default": "regular",
        "figure.figsize": (10.0, 10.0),
        "font.size": 26,
        "axes.labelsize": "medium",
        "axes.unicode_minus": False,
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "legend.fontsize": "small",
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.5,
        "xtick.direction": "in",
        "xtick.major.size": 12,
        "xtick.minor.size": 6,
        "xtick.major.pad": 6,
        "xtick.top": True,
        "xtick.major.top": True,
        "xtick.major.bottom": True,
        "xtick.minor.top": True,
        "xtick.minor.bottom": True,
        "xtick.minor.visible": True,
        "ytick.direction": "in",
        "ytick.major.size": 12,
        "ytick.minor.size": 6.0,
        "ytick.right": True,
        "ytick.major.left": True,
        "ytick.major.right": True,
        "ytick.minor.left": True,
        "ytick.minor.right": True,
        "ytick.minor.visible": True,
        "grid.alpha": 0.8,
        "grid.linestyle": ":",
        "axes.linewidth": 2,
        "savefig.transparent": False,
    }
    plt.style.use(CMS)

    label_names = [
        "H→bb",
        "H→cc",
        "H→gg",
        "H→4q",
        "H→qql",
        "Z→qq",
        "W→qq",
        "t→bqq",
        "t→bl"
    ]

    if args.model == "TransformerSetVAE":
        model = TransformerSetVAE.load_from_checkpoint(args.ckpt).cuda()
    elif args.model == "TransformerClipVAE":
        model = TransformerClipVAE.load_from_checkpoint(args.ckpt).cuda()
    elif args.model == "DeepsetSetVAE":
        model = DeepsetSetVAE.load_from_checkpoint(args.ckpt).cuda()
    elif args.model == "DeepsetClipVAE":
        model = DeepsetClipVAE.load_from_checkpoint(args.ckpt).cuda()
    else:
        raise NotImplementedError(f"model {name} specified is not implemented")
    pred_df = get_predictions(model)

    fig, ax = plt.subplots(figsize = (10, 8), ncols = 1, nrows = 1)
    curves = pd.DataFrame()
    for i, jet_type in zip(range(1, 10), label_names):
        partial_df = pred_df[pred_df['labels'].isin([0, i])]
        fpr, tpr, _ = roc_curve(partial_df["is_signal"], partial_df["scores"])
        auc = roc_auc_score(partial_df["is_signal"], partial_df["scores"])
        fpr, tpr = fpr[(tpr > 0) & (fpr > 0)], tpr[(tpr > 0) & (fpr > 0)]
        curves = pd.concat([
            curves,
            pd.DataFrame({
                "False positive rate": np.linspace(1e-3, 1, 1000),
                "True positive rate": np.interp(np.linspace(1e-3, 1, 1000), fpr, tpr),
                "AUC": auc,
                "Jet type": jet_type,
            })
        ], ignore_index = True)
        print(f"{jet_type}: \t TPR={100*np.interp([0.01], fpr, tpr)[0]:.1f}% \t AUC={auc:.3f}")
    curves = pd.concat([
        curves,
        pd.DataFrame({
            "False positive rate": np.linspace(1e-3, 1, 1000),
            "True positive rate": np.linspace(1e-3, 1, 1000),
            "Jet type": "chance level",
        })
    ], ignore_index = True)
    g = sns.lineplot(data=curves, x="False positive rate", y="True positive rate", hue="Jet type", ax = ax)
    ax.grid(True, which="both", ls="--", color='0.65')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(f"plots/roc_{args.save_name}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()