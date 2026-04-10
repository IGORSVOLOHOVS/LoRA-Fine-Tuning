import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CheburashkaDataset(Dataset):

    def __init__(self, data_dir, instance_prompt, size=512, center_crop=True):
        self.data_dir = data_dir
        self.instance_prompt = instance_prompt
        self.size = size
        self.center_crop = center_crop
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist.")

        self.image_paths = [
            os.path.join(data_dir, file) 
            for file in os.listdir(data_dir) 
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # shifts range from [0, 1] to [-1, 1] as required by SD model
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image = Image.open(self.image_paths[index % self.num_images])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        example["instance_images"] = self.image_transforms(image)
        example["instance_prompt"] = self.instance_prompt
        return example

if __name__ == "__main__":
    dataset = CheburashkaDataset(data_dir="./data/cheburashka", instance_prompt="<cheburashka> plushie")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['instance_images'].shape}")
    print(f"Prompt: {sample['instance_prompt']}")
