# PYTORCH RAY EXAMPLE
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test

def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset

class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_cifar(config, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        print('Using cuda')
        if torch.cuda.device_count() > 1:
            print('Using data parallel')
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            checkpoint=checkpoint,
        )
    print("Finished Training")

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total



def train_mnist(config):
    train_loader, test_loader = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        tune.report(mean_accuracy=acc)

def main():
    if ray.is_initialized():
        print("Shutting down existing Ray instance")
        ray.shutdown()

    print('Initialized ray workers. Number of GPUs: ', ray.get_gpu_ids() )
    ray.init()
    analysis = tune.run(
        train_mnist, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    
# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
#     data_dir = os.path.abspath("./data")
#     load_data(data_dir)
#     config = {
#         "l1": tune.choice([2**i for i in range(9)]),
#         "l2": tune.choice([2**i for i in range(9)]),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16]),
#     }
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2,
#     )
#     print("Prepared configs and scheduler")
#     # if ray.is_initialized():
#     #     ray.shutdown()
#     # ray.init()
#     print("Initialized ray workers")
#     result = tune.run(
#         partial(train_cifar, data_dir=data_dir),
#         resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#     )

#     best_trial = result.get_best_trial("loss", "min", "last")
#     print(f"Best trial config: {best_trial.config}")
#     print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
#     print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

#     best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if gpus_per_trial > 1:
#             best_trained_model = nn.DataParallel(best_trained_model)
#     best_trained_model.to(device)

#     best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
#     best_checkpoint_data = best_checkpoint.to_dict()

#     best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

#     test_acc = test_accuracy(best_trained_model, device)
#     print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    # main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
    main()


# RAY PBT EXAMPLE

# import torch
# import ray
# from ray import air, tune
# from ray.tune.schedulers import PopulationBasedTraining

# from ray.tune.examples.pbt_dcgan_mnist.common import Net
# from ray.tune.examples.pbt_dcgan_mnist.pbt_dcgan_mnist_func import (
#     dcgan_train, download_mnist_cnn
# )

# import matplotlib.pyplot as plt

# # Load the pretrained mnist classification model for inception_score
# mnist_cnn = Net()
# model_path = download_mnist_cnn()
# mnist_cnn.load_state_dict(torch.load(model_path))
# mnist_cnn.to('cuda')
# mnist_cnn.eval()
# # Put the model in Ray object store.
# mnist_model_ref = ray.put(mnist_cnn)

# perturbation_interval = 5
# scheduler = PopulationBasedTraining(
#     perturbation_interval=perturbation_interval,
#     hyperparam_mutations={
#         # Distribution for resampling
#         "netG_lr": tune.uniform(1e-2, 1e-5),
#         "netD_lr": tune.uniform(1e-2, 1e-5),
#     },
# )

# smoke_test = True  # For testing purposes: set this to False to run the full experiment
# tuner = tune.Tuner(
#     dcgan_train,
#     run_config=air.RunConfig(
#         name="pbt_dcgan_mnist_tutorial",
#         stop={"training_iteration": 5 if smoke_test else 150},
#         verbose=1,
#     ),
#     tune_config=tune.TuneConfig(
#         metric="is_score",
#         mode="max",
#         num_samples=2 if smoke_test else 8,
#         scheduler=scheduler,
#     ),
#     param_space={
#         # Define how initial values of the learning rates should be chosen.
#         "netG_lr": tune.choice([0.0001, 0.0002, 0.0005]),
#         "netD_lr": tune.choice([0.0001, 0.0002, 0.0005]),
#         "mnist_model_ref": mnist_model_ref,
#         "checkpoint_interval": perturbation_interval,
#     },
# )
# results_grid = tuner.fit()


# # Uncomment to apply plotting styles
# # !pip install seaborn
# # import seaborn as sns
# # sns.set_style("darkgrid")

# result_dfs = [result.metrics_dataframe for result in results_grid]
# best_result = results_grid.get_best_result(metric="is_score", mode="max")

# plt.figure(figsize=(7, 4))
# for i, df in enumerate(result_dfs):
#     plt.plot(df["is_score"], label=i)
# plt.legend()
# plt.title("Inception Score During Training")
# plt.xlabel("Training Iterations")
# plt.ylabel("Inception Score")
# plt.show()