import torch

from .utils import MODEL_PATH, CLASSES, device

def train(n_epochs, train_dataloader, cnn, optim, criterion, save_model=True):
    best_loss = 100

    for epoch in range(n_epochs):
        running_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optim.step()

            running_loss += loss.item()

            if save_model and loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(cnn.state_dict(), MODEL_PATH)

            if i % 2000 == 1999:
                print(f'Epoch: {epoch} Batch: {i+1} Loss: {running_loss/2000:.3f}')
                running_loss = 0

def test(test_dataloader, cnn):
    correct = {classname: 0 for classname in CLASSES}
    total = {classname: 0 for classname in CLASSES}

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels

            outputs = cnn(inputs)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct[CLASSES[label]] += 1
                total[CLASSES[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct.items():
            accuracy = 100 * float(correct_count) / total[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

