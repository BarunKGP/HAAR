from pathlib import Path
import sys
import os
from omegaconf import DictConfig
import hydra
# import torch
# import torch.distributed as dist

# from torch.distributed import destroy_process_group, init_process_group

from systems.recognition_module import EpicActionRecognitionModule, DDPRecognitionModule
from systems.data_module import EpicActionRecognitionDataModule
from utils import get_device, get_loggers


@hydra.main(config_path="../configs", config_name="pilot_config", version_base=None)
def main(cfg: DictConfig):
    data_module = EpicActionRecognitionDataModule(cfg)
    # debug(data_module.val_loader)
    if cfg.learning.get("ddp", False):
        system = DDPRecognitionModule(cfg)
    else:
        system = EpicActionRecognitionModule(cfg)
    
    # From EPIC script
    # if not cfg.get("log_graph", True):
    #     # MTRN can't be traced due to the model stochasticity so causes a JIT tracer
    #     # error, we allow you to prevent the tracer from running to log the graph when
    #     # the summary writer is created
    #     try:
    #         delattr(system, "example_input_array")
    #     except AttributeError:
    #         pass
    
    system.run_training_loop(data_module, cfg.trainer.max_epochs, Path(cfg.model.save_path))
    sys.exit()  # required to prevent CPU lock (soft bug)

def debug(loader):
    for item in loader:
        rgb, flow = item
        rgb_tensors, rgb_metadata = rgb
        print(len(item))
        print(rgb_tensors.size())
        print(rgb_metadata.keys())
        break


if __name__ == "__main__":
    main()
