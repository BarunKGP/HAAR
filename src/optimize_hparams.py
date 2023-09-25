import ray
from ray import air, tune
from ray.air import session, Checkpoint
from ray.tune.schedulers import PopulationBasedTraining

from tqdm import tqdm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.utils import instantiate

from train import Trainer


initialize(config_path='../configs', version_base=None)
cfg = compose(config_name='pilot_config')

def train_model(config):
    step = 1
    trainer = Trainer("hparams_run", cfg, device=None)
    use_amp = False if cfg.trainer.get("precision", "full") == "full" else True
    optimizer = trainer.optimizer
    accelerator = trainer.accelerator
    model = trainer.model
    loss_fn = trainer.get_loss_fn()
    train_loader = trainer.val_loader  # ? do HPO on val dataset for compute efficiency

    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)

    if session.get_checkpoint():
        checkpoint_dict = session.get_checkpoint().to_dict()
        accelerator.load_state(config.model.save_loc)
        for param_group in optimizer.param_groups:
            if "lr" in config:
                param_group["lr"] = config["lr"]
            if "weight_decay" in config:
                param_group["weight_decay"] = config["weight_decay"]

        last_step = checkpoint_dict["step"]
        step = last_step + 1

    model.train()
    correct_preds = num_elems = loss = 0
    while True:
        for batch in tqdm(train_loader, total=len(train_loader)):
            with accelerator.accumulate(model):
                labels = batch[0][1]["verb_class"]
                logits = model(batch, fp16=use_amp)
                loss += loss_fn(logits, labels).item()
                accurate_preds = trainer.compute_accuracy(logits, labels)
                correct_preds += accurate_preds.long().sum().item()  # type: ignore
                num_elems += accurate_preds.shape[0]  # type: ignore

        checkpoint = None
        if (step + 1) % cfg.trainer.hparams.checkpoint_interval == 0:
            accuracy = correct_preds / num_elems
            accelerator.save_state("hparams")
            checkpoint = Checkpoint.from_dict(
                {
                    "step": step,
                }
            )
            session.report(
                {
                    "mean_accuracy": accuracy,
                    "loss": loss / (step + 1) * len(train_loader),
                    "lr": config["lr"],
                },
                checkpoint=checkpoint,
            )
        step += 1


# Configure and run tuner for PBT
perturbation_interval = 3
scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=perturbation_interval,
    metric="mean_accuracy",
    mode="max",
    hyperparam_mutations={
        "lr": tune.uniform(1e-7, 1e-3),
        "weight_decay": tune.loguniform(1e-7, 1e-3),
    },
)

if ray.is_initialized():
    ray.shutdown()
ray.init()
print("Initialized ray workers")

hparam_cfg = cfg.trainer.hparams
param_space = { 'checkpoint_interval': perturbation_interval }
for key in hparam_cfg.get('param_space', {}):
    param_space[key] = instantiate(hparam_cfg[key])

assert param_space['checkpoint_interval'] == perturbation_interval, "We recommend matching checkpoint_interval with perturbation_interval from the PBT config. This ensures that the PBT algorithm actually exploits the trials in the most recent iteration."

tuner = tune.Tuner(
    train_model,
    run_config=air.RunConfig(
        name="pbt_test",
        stop={"mean_accuracy": hparam_cfg.stop.mean_accuracy, "training_iteration": hparam_cfg.stop.max_epochs},
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=3,
        ),
        storage_path="data/hparam/ray_results",
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=4,
    ),
    param_space=param_space,
    # param_space={
    #     "lr": tune.uniform(1e-6, 1e-3),
    #     "weight_decay": tune.uniform(1e-6, 1e-4),
    #     "checkpoint_interval": perturbation_interval,
    # },
)

results_grid = tuner.fit()

# Plot best results
best_result = results_grid.get_best_result(metric="mean_accuracy", mode="max")
print("Best result logdir:", best_result.log_dir)
print("Best final iteration hyperparameter config:\n", best_result.config)

df = best_result.metrics_dataframe
df = df.drop_duplicate(subset="training_iteration", keep="last")
df.plot("training_iteration", "mean_accuracy")
plt.xlabel("Training iterations (epochs)")
plt.ylabel("Test accuracy")
plt.show
