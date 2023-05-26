import logging
from pathlib import Path
from typing import Union
from omegaconf import DictConfig

from datasets.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset
from datasets.tsn_dataset import TsnDataset
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset, DataLoader
from mp_utils import prepare_distributed_sampler
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
        self.train_gulp_dir = {
            "rgb": Path(cfg.data.rgb.train_gulp_dir),
            "flow": Path(cfg.data.flow.train_gulp_dir),
        }
        self.val_gulp_dir = {
            "rgb": Path(cfg.data.rgb.val_gulp_dir),
            "flow": Path(cfg.data.flow.val_gulp_dir),
        }
        self.test_gulp_dir = {
            "rgb": Path(cfg.data.rgb.test_gulp_dir),
            "flow": Path(cfg.data.flow.test_gulp_dir),
        }
        self.ddp = cfg.trainer.get("ddp", False)
        self.cfg = cfg
        self.rgb_train_transform, self.rgb_test_transform = self.get_transforms(
            "rgb", cfg.data.rgb
        )
        self.flow_train_transform, self.flow_test_transform = self.get_transforms(
            "flow", cfg.data.flow
        )

    def get_transforms(self, modality, cfg) -> "tuple[Compose, Compose]":
        """Configure image augmentations for frames

        Returns:
            tuple[Compose, Compose]: train_transforms, test_transforms
        """
        channel_count = 3 if modality == "rgb" else 2 * cfg.segment_length
        common_transforms = Compose(
            [
                Stack(bgr=modality == "rgb" and cfg.preprocessing.get("bgr", False)),
                ToTorchFormatTensor(div=cfg.preprocessing.rescale),
                GroupNormalize(
                    mean=list(cfg.preprocessing.mean),
                    std=list(cfg.preprocessing.std),
                ),
                ExtractTimeFromChannel(channel_count),
            ]
        )
        train_transforms = Compose(
            [
                GroupMultiScaleCrop(
                    cfg.preprocessing.input_size,
                    cfg.train_augmentation.multiscale_crop_scales,
                ),
                GroupRandomHorizontalFlip(is_flow=modality == "flow"),
                common_transforms,
            ]
        )
        test_transforms = Compose(
            [
                GroupScale(cfg.test_augmentation.rescale_size),
                GroupCenterCrop(cfg.preprocessing.input_size),
                common_transforms,
            ]
        )

        return train_transforms, test_transforms

    def train_dataloader(self, rank: Union[None, int] = None):
        frame_count = self.cfg.data.frame_count
        LOG.info(f"Training dataset: frame count {frame_count}")

        rgb_dataset = TsnDataset(
            self._get_video_dataset(self.train_gulp_dir["rgb"], modality="rgb"),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.rgb_train_transform,
        )
        flow_dataset = TsnDataset(
            self._get_video_dataset(self.train_gulp_dir["flow"], modality="flow"),
            num_segments=frame_count,
            segment_length=self.cfg.data.segment_length,
            transform=self.flow_train_transform,
        )
        if self.cfg.data.get("train_on_val", False):
            LOG.info("Training on training set + validation set")
            rgb_dataset = ConcatDataset(
                [
                    rgb_dataset,
                    TsnDataset(
                        self._get_video_dataset(
                            self.val_gulp_dir["rgb"], modality="rgb"
                        ),
                        num_segments=frame_count,
                        segment_length=self.cfg.data.segment_length,
                        transform=self.rgb_train_transform,
                    ),
                ]
            )
            flow_dataset = ConcatDataset(
                [
                    flow_dataset,
                    TsnDataset(
                        self._get_video_dataset(
                            self.val_gulp_dir["flow"], modality="flow"
                        ),
                        num_segments=frame_count,
                        segment_length=self.cfg.data.segment_length,
                        transform=self.flow_train_transform,
                    ),
                ]
            )
        dataset = ConcatDataset([rgb_dataset, flow_dataset])
        LOG.info(f"Training dataset size: {len(dataset)}")

        if self.ddp:
            assert rank is not None, "rank must be specified for DDP."
            return prepare_distributed_sampler(
                dataset=dataset,
                rank=rank,
                world_size=self.cfg.learning.ddp.world_size,
                batch_size=self.cfg.learning.batch_size,
                num_workers=self.cfg.data.worker_count,
                pin_memory=self.cfg.data.pin_memory,
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=False,  # ? should shuffle be true
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def val_dataloader(self, rank: Union[None, int] = None):
        frame_count = self.cfg.data.frame_count
        LOG.info(f"Validation dataset: frame count {frame_count}")
        dataset = ConcatDataset(
            [
                TsnDataset(
                    self._get_video_dataset(self.val_gulp_dir["rgb"], modality="rgb"),
                    num_segments=frame_count,
                    segment_length=self.cfg.data.segment_length,
                    transform=self.rgb_test_transform,
                    test_mode=True,
                ),
                TsnDataset(
                    self._get_video_dataset(self.val_gulp_dir["flow"], modality="flow"),
                    num_segments=frame_count,
                    segment_length=self.cfg.data.segment_length,
                    transform=self.flow_test_transform,
                    test_mode=True,
                ),
            ]
        )
        LOG.info(f"Validation dataset size: {len(dataset)}")

        # if self.ddp:
        #     assert rank is not None, "rank must be specified for DDP."
        #     return prepare_distributed_sampler(
        #         dataset=dataset,
        #         rank=rank,
        #         world_size=self.cfg.learning.ddp.world_size,
        #         batch_size=self.cfg.learning.batch_size,
        #         num_workers=self.cfg.data.worker_count,
        #         pin_memory=self.cfg.data.pin_memory,
        #     )
        return DataLoader(
            dataset=dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def test_dataloader(self, rank: Union[None, int] = None):
        frame_count = self.cfg.data.get("test_frame_count", self.cfg.data.frame_count)
        LOG.info(f"Test dataset: frame count {frame_count}")
        dataset = ConcatDataset(
            [
                TsnDataset(
                    self._get_video_dataset(self.test_gulp_dir["rgb"], modality="rgb"),
                    num_segments=frame_count,
                    segment_length=self.cfg.data.segment_length,
                    transform=self.rgb_test_transform,
                    test_mode=True,
                ),
                TsnDataset(
                    self._get_video_dataset(
                        self.test_gulp_dir["flow"], modality="flow"
                    ),
                    num_segments=frame_count,
                    segment_length=self.cfg.data.segment_length,
                    transform=self.flow_test_transform,
                    test_mode=True,
                ),
            ]
        )

        LOG.info(f"Test dataset size: {len(dataset)}")

        if self.ddp:
            assert rank is not None, "rank must be specified for DDP."
            return prepare_distributed_sampler(
                dataset=dataset,
                rank=rank,
                world_size=self.cfg.learning.ddp.world_size,
                batch_size=self.cfg.learning.batch_size,
                num_workers=self.cfg.data.worker_count,
                pin_memory=self.cfg.data.pin_memory,
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self.cfg.learning.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.worker_count,
            pin_memory=self.cfg.data.pin_memory,
        )

    def _get_video_dataset(self, gulp_dir_path, modality):
        if modality == "rgb":
            return EpicVideoDataset(gulp_dir_path, drop_problematic_metadata=True)
        elif modality == "flow":
            return EpicVideoFlowDataset(gulp_dir_path, drop_problematic_metadata=True)
        else:
            raise ValueError(f"Unknown modality {modality!r}")
