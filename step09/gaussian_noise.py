import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def reverse_to_img(x):
    x = x*255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

# q(x_t | x_0)
def add_noise(x_0, t, betas):
    T = len(betas)
    assert t >= 1 and t <= T
    
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x_0)
    xt = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1-alpha_bar) * eps
    return xt

image = plt.imread("step09/flower.png")
print(image.shape)
preprocess = transforms.ToTensor()
x = preprocess(image)
print(x.shape)

T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)

t = 100
x_t = add_noise(x, t, betas)
img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f"Noise: {t}")
plt.axis('off')
plt.show()