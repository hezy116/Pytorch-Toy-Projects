import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 64
image_size = [1, 28, 28]
batch_size = 128

'''
Generator and Discriminator
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(latent_dim, 256), 
                                   nn.BatchNorm1d(256), 
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(256, 512), 
                                   nn.BatchNorm1d(512), 
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(512, 1024), 
                                   nn.BatchNorm1d(1024), 
                                   nn.ReLU())
        self.conv4 = nn.Linear(1024, np.prod(image_size, dtype=np.int32))
    
    def forward(self, z):
        # shape of z: [batch_size, latent_dim]
        
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = torch.tanh(self.conv4(z))

        return z.reshape(z.shape[0], *image_size)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(np.prod(image_size), 512), 
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Linear(512, 256), 
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Linear(256, 128), 
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Linear(128, 1)
    
    def forward(self, img):
        # reshape img to [batch_size, flatten_size], where flatten_size should be 1 * 28 * 28
        img = img.reshape(img.shape[0], -1)

        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = self.conv4(img)

        return F.sigmoid(img)


'''
Loss Visualize
'''
def plot_loss(d_losses, g_losses, save_path=None):
    """
    plot loss figures
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator / Discriminator Loss")
    plt.plot(g_losses, label="G Loss", alpha=0.7)
    plt.plot(d_losses, label="D Loss", alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
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

G = Generator()
D = Discriminator()
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)


epochs = 50
d_losses = []
g_losses = []
for epoch in range(epochs):
    print(f"\n___epoch: {epoch}___")
    for i, minibatch in enumerate(dataloader):
        data_imgs, labels = minibatch

        # G-step
        # Using 'Flip Trick', which use max{log(D(G(z)))} instead of min{log(1-D(G(z)))}
        z = torch.randn(batch_size, latent_dim)
        pred_imgs = G(z)

        G_optimizer.zero_grad()
        G_loss = loss_fn(D(pred_imgs), labels_one)
        G_loss.backward()
        G_optimizer.step()

        # D-step
        data_loss = loss_fn(D(data_imgs), labels_one)
        gen_loss = loss_fn(D(pred_imgs.detach()), labels_zero)

        D_optimizer.zero_grad()
        D_loss = data_loss + gen_loss
        D_loss.backward()
        D_optimizer.step()

        if i % 500 == 0:
            print(f"step:{len(dataloader) * epoch + i}, data_loss:{data_loss.item()}, gen_loss:{gen_loss.item()}")
            d_losses.append(data_loss.item())
            g_losses.append(gen_loss.item())
            image = pred_imgs[:16].data
            torchvision.utils.save_image(image, f"./MNIST_generate/image_{len(dataloader) * epoch + i}.png", nrow=4, normalize=True)

        if i % 2000 == 0:
            plot_loss(d_losses, g_losses, f"./loss_figures/loss_figure_{len(dataloader) * epoch + i}")