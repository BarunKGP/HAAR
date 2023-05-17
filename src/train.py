import logging
import sys
from logging.handlers import RotatingFileHandler

import torch.nn as nn
from constants import (
    NOUN_CLASSES,
    VERB_CLASSES,
)
from systems.trainer import Trainer
from utils import get_loggers

# LOGGING
stream_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = RotatingFileHandler(
    filename="data/pilot-01/logs/train.log", maxBytes=50000, backupCount=5
)
logger = get_loggers(
    name=__name__,
    handlers=[(stream_handler, logging.INFO), (file_handler, logging.ERROR)],
)

if __name__ == "__main__":
    trainer = Trainer(
        VERB_CLASSES,
        NOUN_CLASSES,
        nn.CrossEntropyLoss(reduction="mean"),
        save_path="data/pilot-01",
    )
    trainer.training_loop(num_epochs=1)
