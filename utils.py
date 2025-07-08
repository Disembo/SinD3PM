import torchvision
import yaml
from PIL import Image


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def build_batch(config: dict):
    """
    Build a batch of images from the given configuration.

    Returns:
        A tensor of shape (batch_size, 3, image_size, image_size) containing the images.
    """
    image = Image.open(config['image_path']).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config['image_size'], config['image_size'])),
        torchvision.transforms.ToTensor()  # scaled to [0, 1]
    ])
    image = transforms(image)
    image = image.unsqueeze(0)
    batch_size = config['batch_size']
    if batch_size > 1:
        image = image.repeat(batch_size, 1, 1, 1)  # Repeat the image for batch size
    return image
