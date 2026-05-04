import torch
import torchvision
import matplotlib.pyplot as plt
from UNet import ResUNet
from ddpm import p_sample_loop

def run_inference(checkpoint_path, device="cuda"):
    """
    展示64张通过去噪过程生成的图片
    """
    model = ResUNet().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    shape = (64, 1, 32, 32)
    with torch.no_grad():
        final_img, _ = p_sample_loop(model, shape)
    
    final_img = (final_img.clamp(-1, 1) + 1) / 2
    grid = torchvision.utils.make_grid(final_img, nrow=8)
    
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig("./generate/inference_result.png")

def plot_denoising_process(checkpoint_path, device="cuda"):
    """
    展示一张图片从纯噪声到清晰图片的去噪过程
    """
    model = ResUNet().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    x_final, imgs = p_sample_loop(model, shape=(8, 1, 32, 32))

    num_stages = len(imgs)
    fig, axes = plt.subplots(1, num_stages, figsize=(15, 3))
    for i, img_tensor in enumerate(imgs):
        img = img_tensor[0]
        # Denormalize, [-1, 1] -> [0, 1]
        img = (img + 1.0) / 2.0
        img = torch.clamp(img, 0.0, 1.0)        
        # (C, H, W) -> (H, W, C), to fit Matplotlib
        img = img.permute(1, 2, 0).numpy()

        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
            
        axes[i].axis('off')
        
        if i == num_stages - 1:
            axes[i].set_title("Final")
        else:
            axes[i].set_title(f"Stage {i+1}")

    plt.tight_layout()
    plt.savefig("./generate/denoising_process.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference("./checkpoint.pth", device)
    plot_denoising_process("./checkpoint.pth", device)