# -Image-to-Image-Translation-with-cGAN-pix2pix-.
pip install torch torchvision matplotlib
import torch
from torchvision.utils import make_grid
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os

# Function to show tensor image
def show_tensor_image(image_tensor):
    image = image_tensor.detach().cpu()
    grid = make_grid(image, normalize=True)
    np_img = grid.permute(1, 2, 0).numpy()
    plt.imshow(np_img)
    plt.axis("off")
    plt.show()

# Dummy U-Net Generator for example (replace with trained model)
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load an image (edge map or sketch)
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)

input_image = load_image("input_edge.jpg").to(device)
output_image = gen(input_image)

# Display
show_tensor_image(input_image)
show_tensor_image(output_image)
