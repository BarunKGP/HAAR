from pathlib import Path
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
from mp_utils import ddp_setup
from torch.distributed import destroy_process_group

from systems.recognition_module import (
    EpicActionRecognitionDataModule,
    EpicActionRecognitionModule,
)
from utils import get_loggers

LOG = get_loggers(name=__name__, filename="data/pilot-01/logs/train.log")


@hydra.main(config_path="../configs", config_name="pilot_config", version_base=None)
def main(cfg: DictConfig):
    LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    data_module = EpicActionRecognitionDataModule(cfg)
    LOG.debug("EpicActionRecognitionDataModule initialized")

    # debug
    # loader = data_module.val_dataloader()
    # for item in loader:
    #     rgb, flow = item
    #     rgb_tensors, rgb_metadata = rgb
    #     # print(len(item))
    #     print(rgb_tensors.size())
    #     print(rgb_metadata.keys())
    #     break

    system = EpicActionRecognitionModule(cfg, data_module)
    LOG.debug("EpicActionRecognitionSystem initialized")
    if not cfg.get("log_graph", True):
        # MTRN can't be traced due to the model stochasticity so causes a JIT tracer
        # error, we allow you to prevent the tracer from running to log the graph when
        # the summary writer is created
        try:
            delattr(system, "example_input_array")
        except AttributeError:
            pass
    ddp = cfg.trainer.get("ddp", False)
    if ddp:
        ddp_setup()
    # LOG.info("Starting training....")
    system.run_training_loop(1, Path(cfg.model.save_path))
    if ddp:
        destroy_process_group()
    LOG.info("Training completed!")
    sys.exit(0)  # required to prevent CPU lock (soft bug)


if __name__ == "__main__":
    main()
