import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import timm  # if using a pretrained GeM-compatible model
from natsort import natsorted

class GeMFeatureExtractor:
    def __init__(self, model_name="resnet50_gem", image_dir="path_to_images", device="cuda"):
        self.device = device
        self.image_dir = image_dir
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()

        # Preprocessing transform to match model input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # or model.default_cfg['input_size']
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def compute_gem_features(self):
        """
        Compute GeM global descriptors for all images in the image directory.
        Returns:
            descriptors: (N, D) numpy array
            image_files: list of filenames in the same order
        """
        files = natsorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg") or f.endswith(".png")])
        descriptors = []
        image_files = []

        for fname in tqdm(files, desc="Extracting GeM descriptors"):
            path = os.path.join(self.image_dir, fname)
            image = Image.open(path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                desc = self.model.forward_features(input_tensor).squeeze().cpu().numpy()

            descriptors.append(desc)
            image_files.append(fname)

        return np.stack(descriptors), image_files