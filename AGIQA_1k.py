import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import torchvision.transforms as transforms

class AGIQA1kDataset(Dataset):
    def __init__(self, csv_file, img_dir, dtype=torch.float32):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.dtype = dtype
        
        # 只保留一个转换，使用224x224的标准配置
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        prompt = self.data.iloc[idx, 1]
        score = torch.tensor(self.data.iloc[idx, 2], dtype=self.dtype)

        # 应用转换
        image = self.transform(image).view(1, 3, 512, 512)
        
        return {
            'image': image,
            'prompt': prompt,
            'score': score
        }

def custom_collate(batch, dtype=torch.float32):
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    scores = torch.tensor([item['score'] for item in batch], dtype=dtype)
    
    return {
        'image': images,
        'prompt': prompts,
        'score': scores
    }