import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from UNet import ResUNet

"""
MNIST dataloader
"""
def get_mnist_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = datasets.MNIST(
        root="./MNIST_data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    return dataloader


"""
Forward Process
"""
T = 1200

def linear_beta_schedule(timesteps):
    start = 0.0001
    end = 0.02
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(T)
alphas = 1. - betas

alphas_cumprod = torch.cumprod(alphas, dim=0) # \bar{\alpha_t}
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_betas_cumprod = torch.sqrt(1. - alphas_cumprod)

def extract(a, t, x_shape):
    """
    Extract a[t] from pre-calculated 1D torch.tensor a,
    and reshape it so it can multiply x_shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_0, t, epsilon):
    """
    Sample x_t using x_0, t and epsilon(noise)
    """
    if epsilon == None:
        epsilon = torch.rand_like(x_0)
    
    sqrt_alpha_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_beta_cumprod_t = extract(sqrt_betas_cumprod, t, x_0.shape)

    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_beta_cumprod_t * epsilon
    return x_t


"""
Reverse Process
"""
sqrt_recip_alphas = torch.sqrt(1. / alphas)
sqrt_betas = torch.sqrt(betas)

@torch.no_grad()
def p_sample(model, x_t, t):
    beta_t = extract(betas, t, x_t.shape)
    sqrt_beta_t = extract(sqrt_betas, t, x_t.shape)
    sqrt_beta_cumprod_t = extract(sqrt_betas_cumprod, t, x_t.shape)
    sqrt_recip_alpha_t = extract(sqrt_recip_alphas, t, x_t.shape)

    pred_epsilon = model(x_t, t)
    mean = sqrt_recip_alpha_t * (x_t - beta_t * pred_epsilon / sqrt_beta_cumprod_t)
    z = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
    sample = mean + sqrt_beta_t * z

    return sample

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    batch_size = shape[0]    
    
    x_t = torch.randn(shape, device=device)
    imgs = []
    for i in reversed(range(0, T)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)        
        x_t = p_sample(model, x_t, t)

        if i % 100 == 0 or i == 0:
            imgs.append(x_t.cpu())
            
    return x_t, imgs



if __name__ == "__main__":
    batch_size = 64
    dataloader = get_mnist_dataloader(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    epochs = 100
    model = ResUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("==== Start Training ====\n")
    
    start_epoch = 0
    checkpoint_path = "checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint {checkpoint_path} detected, epochs 1~{start_epoch} have been trained, ready to continue...")

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.
        
        for step, (images, labels) in enumerate(dataloader):
            x_0 = images.to(device)
            batch_size_now = x_0.shape[0]

            t = torch.randint(0, T, (batch_size_now, ), device=device)
            epsilon = torch.randn_like(x_0).to(device)
            
            x_t = q_sample(x_0, t, epsilon)
            pred_epsilon = model(x_t, t)
            loss = loss_fn(pred_epsilon, epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")
                    
        avg_loss = epoch_loss / len(dataloader)
        print(f"==== Epoch {epoch+1} Finished! Average Loss: {avg_loss:.4f} ====\n")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, "checkpoint.pth")