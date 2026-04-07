import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 64
image_size = [1, 28, 28]
batch_size = 64

'''
CNN
'''
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()

        # image_shape = [batch_size, 1, 28, 28]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
        )
        # image_shape = [batch_size, 16, 14, 14]
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),
        )
        # image_shape = [batch_size, 64, 7, 7]

        self.fc1 = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(), 
        )
        
        # nn.CrossEntropyLoss 自带归一化，所以这里不需要再 softmax
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


'''
Visualization
'''
def plot_loss(losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.title("Model Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()

def show_pred(images, outputs, labels, nrow=4, ncol=4, save_path=None):
    fig, axes = plt.subplots(nrow, ncol, figsize=(7, 7))
    for idx, ax in enumerate(axes.flatten()):
        img = images[idx].squeeze().cpu().numpy()
        ax.imshow(img, cmap='gray')
        pred, label = torch.argmax(outputs[idx]), labels[idx]
        color = "g" if pred == label else "r"
        ax.set_title(f"pred:{pred}, label:{label}", color=color)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"prediction figure saved in {save_path}")
    else:
        plt.show()


'''
Training
'''
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),                      # pixel value \in [0, 1]
    transforms.Normalize((0.5, ), (0.5, )),     # stretch pixel value range to [-1, 1]
])
dataset = torchvision.datasets.MNIST("./MNIST_data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

testset = torchvision.datasets.MNIST("./MNIST_data", train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True, drop_last=True)


cnn_model = simpleCNN()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters())
loss_fn = nn.CrossEntropyLoss()


def calc_correct_rate():
    cnn_model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = cnn_model(images)
            pred = torch.argmax(outputs, dim=1)

            total += len(labels)
            correct += (pred == labels).sum().item()

    print(f"correct = {correct}, total = {total}")

    cnn_model.train()
    return 1.0 * correct / total
    

epochs = 50
cnn_model.train()
losses = []
for epoch in range(epochs):
    print(f"\n___epoch: {epoch}___")
    for i, minibatch in enumerate(dataloader):
        images, labels = minibatch

        outputs = cnn_model(images)
        target = F.one_hot(labels, 10).float()
        loss = loss_fn(outputs, labels)

        cnn_optimizer.zero_grad()
        loss.backward()
        cnn_optimizer.step()

        if i % 250 == 0:
            losses.append(loss.item())

        if i % 500 == 0:
            print(f"i = {i}, loss = {loss.item():.4f}")
            show_pred(images[:16], outputs[:16], labels[:16], 4, 4, "./prediction_figure.png")
            plot_loss(losses, "./loss_figure.png")

    print(f"correct_rate = {calc_correct_rate()}")