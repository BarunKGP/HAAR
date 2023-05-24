from omegaconf import DictConfig, OmegaConf
import hydra

from systems.recognition_module import (
    EpicActionRecognitionDataModule,
    EpicActionRecognitionModule,
)
from utils import get_loggers

LOG = get_loggers(name=__name__, filename="data/pilot-01/logs/train.log")


@hydra.main(config_path="../configs", config_name="pilot_config")
def main(cfg: DictConfig):
    LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    data_module = EpicActionRecognitionDataModule(cfg)
    LOG.debug("EpicActionRecognitionDataModule initialized")
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
    LOG.info("Starting training....")
    system.training_loop(1, cfg.model.save_path)
    LOG.info("Training completed!")


if __name__ == "__main__":
    main()
