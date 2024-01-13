import torch
import torch.optim as optim
import torch.nn as nn

from .dataloaders import get_dataloaders
from .models import CNN
from .utils import MODEL_PATH, device
from .train import train, test

def main(lr=0.001, epochs=10, load_model=True):
    train_dataloader, test_dataloader = get_dataloaders()
    cnn = CNN().to(device)
    if load_model:
        cnn.load_state_dict(torch.load(MODEL_PATH))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum=0.9)

    train(epochs, train_dataloader, cnn, optimizer, criterion, save_model=True)
    test(test_dataloader, cnn)

main()