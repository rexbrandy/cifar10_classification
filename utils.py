import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MODEL_PATH = './outputs/cifar_net.pth'

def visualize_data(training_data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, 1
    for i in range(1, cols * rows + 1):

        sample_idx = torch.randint(len(training_data), size=(1,)).item()

        img, label = training_data[sample_idx]

        figure.add_subplot(rows, cols, i)
        plt.title(CLASSES[label])
        plt.axis("off")

        img = img / 2 + 0.5
        np_img = np.transpose(img, (1, 2, 0))
        plt.imshow(np_img)
    plt.show()

def show_predictions(test_dataloader):
    images, labels = next(iter(test_dataloader))
    
    img = torchvision.utils.make_grid(images)

    print('GroundTruth: ', ' '.join(f'{CLASSES[labels[j]]:5s}' for j in range(5)))
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()