import ray
from ray import air, tune
from ray.air import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining

from models.models import HaarModel
from systems.data_module import EpicActionRecognitionDataModule

def train_model(config):
    model = HaarModel(config, config.model.transformer.dropout, None, config.model.linear_out)
    datamodule = EpicActionRecognitionDataModule(config)
    # trainer = 

