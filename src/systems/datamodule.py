import logging
from pathlib import Path
from omegaconf import DictConfig

from datasets.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset
from datasets.tsn_dataset import TsnDataset
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset, DataLoader
from transforms import (
    ExtractTimeFromChannel,
    GroupCenterCrop,
    GroupMultiScaleCrop,
    GroupNormalize,
    GroupRandomHorizontalFlip,
    GroupScale,
    Stack,
    ToTorchFormatTensor,
)

LOG = logging.getLogger(
    __name__
)  # TODO: update to mp logging using file and stream handlers


class EpicActionRecognitionDataModule(object):
    def __init__(self, cfg: DictConfig) -> None:
        self.train_gulp_dir = Path(cfg.data.train_gulp_dir)
        self.val_gulp_dir = Path(cfg.data.val_gulp_dir)
        self.test_gulp_dir = Path(cfg.data.test_gulp_dir)
        self.cfg = cfg
        self.train_transform, self.test_transform = self.get_transforms()

    def get_transforms(self) -> "tuple[Compose, Compose]":
        """Configure image augmentations for frames

        Returns:
            tuple[Compose, Compose]: train_transforms, test_transforms
        """
        channel_count = (
            3 if self.cfg.modality == "RGB" else 2 * self.cfg.data.segment_length
        )
        common_transforms = Compose(
            [
                Stack(
                    bgr=self.cfg.modality == "RGB"
                    and self.cfg.data.preprocessing.get("bgr", False)
                ),
                ToTorchFormatTensor(div=self.cfg.data.preprocessing.rescale),
                GroupNormalize(
                    mean=list(self.cfg.data.preprocessing.mean),
                    std=list(self.cfg.data.preprocessing.std),
                ),
                ExtractTimeFromChannel(channel_count),
            ]
        )
        train_transforms = Compose(
            [
                GroupMultiScaleCrop(
                    self.cfg.data.preprocessing.input_size,
                    self.cfg.data.train_augmentation.multiscale_crop_scales,
                ),
                GroupRandomHorizontalFlip(is_flow=self.cfg.modality == "Flow"),
                common_transforms,
            ]
        )
        test_transforms = Compose(
            [
                GroupScale(self.cfg.data.test_augmentation.rescale_size),
                GroupCenterCrop(self.cfg.data.preprocessing.input_size),
                common_transforms,
            ]
        )

        return train_transforms, test_transforms

    def train_dataloader(self):
        frame_count = self.cfg.data.frame_count
        LOG.info(f"Training dataset: frame count {frame_count}")
        dataset = TsnDataset(
            self._get_video_dataset(self.train_gulp_dir),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.train_transform,
        )
        if self.cfg.data.get("train_on_val", False):
            LOG.info("Training on training set + validation set")
            dataset = ConcatDataset(
                [
                    dataset,
                    TsnDataset(
                        self._get_video_dataset(self.val_gulp_dir),
                        num_segments=frame_count,
                        segment_length=self.cfg.data.segment_length,
                        transform=self.train_transform,
                    ),
                ]
            )
        LOG.info(f"Training dataset size: {len(dataset)}")

        return DataLoader(
            dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def val_dataloader(self):
        frame_count = self.cfg.data.frame_count
        LOG.info(f"Validation dataset: frame count {frame_count}")
        dataset = TsnDataset(
            self._get_video_dataset(self.val_gulp_dir),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.test_transform,
            test_mode=True,
        )
        LOG.info(f"Validation dataset size: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def test_dataloader(self):
        frame_count = self.cfg.data.get("test_frame_count", self.cfg.data.frame_count)
        LOG.info(f"Test dataset: frame count {frame_count}")
        dataset = TsnDataset(
            self._get_video_dataset(self.test_gulp_dir),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.test_transform,
            test_mode=True,
        )
        LOG.info(f"Test dataset size: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def _get_video_dataset(self, gulp_dir_path):
        if self.cfg.modality.lower() == "rgb":
            return EpicVideoDataset(gulp_dir_path, drop_problematic_metadata=True)
        elif self.cfg.modality.lower() == "flow":
            return EpicVideoFlowDataset(gulp_dir_path, drop_problematic_metadata=True)
        else:
            raise ValueError(f"Unknown modality {self.cfg.modality!r}")
