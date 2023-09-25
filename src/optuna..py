import optuna
from hydra import compose, initialize
from omegaconf import OmegaConf

from train import get_device_from_config


def objective(trial):
    heads = trial.suggest_int('nhead', 4, 10)
    batch_size = trial.suggest_int('batch_size', 4, 16)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    if optimizer == 'SGD':
        optim_arg = trial.suggest_float('optim_arg', 0.1, 0.9)  # momentum
        arg_key = "momentum"
    else:
        optim_arg = trial.suggest_float('optim_arg', 1e-5, 1e-3, log=True)  # weight_decay
        arg_key = "weight_decay"

    lr = trial.suggest_float('lr', 1e-5, 0.01, log=True)
    attention_dropout = trial.suggest_float('attention_dropout', 0.1, 0.7)

    initialize(config_path='configs', job_name='hparam_tune')
    cfg = compose(config_name='pilot_config.yaml', overrides=[
        f"model.transformer.nhead={heads}",
        f"model.transformer.dropout={attention_dropout}",
        f"learning.batch_size={batch_size}",
        f"learning.optimizer.type={optimizer}",
        f"learning.optimizer.args.{arg_key}={optim_arg}",
        f"learning.lr={lr}"
    ])

    # LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    device = get_device_from_config(cfg)
    trainer = Trainer(name="transformers_testing", cfg=cfg, device=device)  # type: ignore
    trainer.train(10)
    # sys.exit()

def main():
    study = optuna.create_study()


    