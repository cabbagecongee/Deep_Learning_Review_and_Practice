#test data.py
from data import get_dataloader
import yaml

with open("config.yaml") as f:
    data_cfg = yaml.safe_load(f)["data"]

data_cfg.update({
    "basedir":     "./data",
    "batch_size":   2,
    "max_files":    1,
    "target_class": None,
})


dl = get_dataloader(
    split        = data_cfg["split"],
    batch_size   = data_cfg["batch_size"],
    shuffle      = True,                   # or data_cfg.get("shuffle", True)
    basedir      = data_cfg["basedir"],
    max_files    = data_cfg["max_files"],
    target_class = data_cfg["target_class"],
)

batch = next(iter(dl))

# 4) inspect
print("particles:", batch["particles"].shape)   # → (2, 50, feat_dim)
print("jet_feats:", batch["jets"].shape) # → (2, jet_feat_dim)
print("labels:", batch["labels"].shape)  # → (2,)

# peek at a few values
print("first particle vec:", batch["particles"][0, :5].numpy())
print("first jet feature:", batch["jets"][0, :5].numpy())
print("first labels:", batch["labels"].numpy())