#eval.py
import torch
import yaml
import numpy as np

from data import get_dataloader
from model import build_model, load_pretrained_model
from utils import compute_roc_auc, compute_rejection, plot_roc, plot_rejection

def main():
    full_cfg   = yaml.safe_load(open("config.yaml"))
    data_cfg   = full_cfg["data"]
    model_cfg  = full_cfg["model"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = get_dataloader(
        split        = data_cfg["split"],
        batch_size   = data_cfg["batch_size"],
        shuffle      = True,
        basedir      = data_cfg["basedir"],
        max_files    = data_cfg["max_files"],
        target_class = data_cfg["target_class"],
    )

    model = build_model(model_cfg)
    model = load_pretrained_model(model, model_cfg["ckpt_path"], device=device)
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            pf     = batch["particles"].to(device)
            mask   = (pf.abs().sum(dim=2)>0).float().to(device)
            # assume your 4-vector = same channels or stored in `batch["jets"]`
            v      = batch["jets"].unsqueeze(1).expand(-1, pf.size(1), -1).to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(pf, v, mask)            # [B, num_classes]
            probs  = torch.softmax(logits, dim=1)
            scores = probs[:, data_cfg["target_class"]].cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, auc_score = compute_roc_auc(all_labels, all_scores, data_cfg["target_class"])
    rej, eff            = compute_rejection(fpr, tpr)
    print(f"AUC = {auc_score:.4f}")
    plot_roc(fpr, tpr, auc_score,   save_to="roc_curve.png")
    plot_rejection(eff, rej, save_to="rej_vs_eff.png")

if __name__ == "__main__":
    main()
