import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler

from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize

from accelerate import Accelerator
from deepspeed.runtime.utils import see_memory_usage

class TestModel(nn.Module):
    def __init__(self, dropout, device=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(), 
            nn.Linear(8*11*11, 10)
        )
        self.device = device

    def forward(self, x):
        return self.model(x)
    
def main():
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ]
        )
    )
    train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    
    model = TestModel(dropout=0.2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-3)
    # scaler = GradScaler()
    
    accelerator = Accelerator(device_placement=True)
    accelerator.print("Accelerator initialized")
    train_loader, test_loader, model, optimizer = accelerator.prepare(
        train_loader, test_loader, model, optimizer
    )
    for epochs in range(4):
        model.train()
        running_loss = 0.
        for batch in train_loader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                features, labels = batch
                # with autocast():
                logits = model(features)
                loss = loss_fn(logits, labels)
                
                running_loss += loss.item()
                
                # scaler.scale(loss)
                accelerator.backward(loss)
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
        
        model.eval()
        val_loss = 0.
        num_elems = 0
        correct = 0
        for features, labels  in test_loader:
            with torch.no_grad():
                logits = model(features)
            preds = nn.Softmax(dim=-1)(logits).argmax(dim=-1)
            accurate_preds = accelerator.gather(preds) == accelerator.gather(labels) 
            num_elems += accurate_preds.shape[0] # type: ignore
            correct += accurate_preds.long().sum().item() # type: ignore
            val_loss += F.cross_entropy(logits, labels, reduction='mean').item()

        accelerator.print(f'\nTraining loss @ epoch {epochs + 1} = {running_loss/len(train_loader)}')
        accelerator.print(f'Validation loss = {val_loss / len(test_loader)}')
        accelerator.print(f'Validation accuracy = {correct / num_elems}')


    accelerator.print("Finished training!")
    return


            

if __name__ == '__main__':
    main()

