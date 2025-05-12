import pytorch_lightning as pl
from torch.utils.data import DataLoader
from threestudio import register
from threestudio.utils.config import parse_structured
from your_package.datasets import ShapenetChairsModified  # wherever you put it
from threestudio.datamodules.random_multiview_camera import RandomMultiviewCameraDataModuleConfig

def collate_fn(batch):
    # batch is a list of length 1 (since batch_size=1)
    b = batch[0]
    out = {}
    for k, v in b.items():
        # if tensor and has a view‐dim, merge it into batch‐dim
        if isinstance(v, torch.Tensor) and v.ndim > 3:
            out[k] = v.reshape((v.shape[0],) + v.shape[1:])
        else:
            out[k] = v
    return out


@register("shapenet-multiview-camera-datamodule")
class ShapeNetMultiviewCameraDataModule(pl.LightningDataModule):
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg):
        super().__init__()
        # you can reuse the same config class or make your own if you need new fields
        self.cfg = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            # pass your config into your dataset
            base_dataset = ShapenetChairsModified(self.cfg, dataset_name="train")
        self.train_dataset = ShapenetChairsAsRaysDataset(base_dataset)
        if stage in (None, "fit", "validate"):
            base_dataset = ShapenetChairsModified(self.cfg, dataset_name="test")
        self.val_dataset = ShapenetChairsAsRaysDataset(base_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size[0],  # or however you want to index it
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=False,
        )
