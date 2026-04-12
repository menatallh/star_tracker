import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from PIL import Image
import os
import json
import pandas as pd
from helpers import *
import random
from torch.utils.data import ConcatDataset, WeightedRandomSampler


def replace_image_path(original_path):
    #new_path = original_path.replace('/processed-imagesp2/', 'processed-imagesp2/')
    old_path='processed-imagesp2'
    new_path=original_path.split('/')[-1]
    #print(original_path.split('/')[1:])
    return os.path.join(old_path,new_path)
def  join_path(original_path):
         new_path='/content/processed-imagesp_new/'
         return  os.path.join(new_path,original_path)


import torch

def encode_labels(ra_deg, dec_deg):
    # 1) convert to radians
    ra_rad  = torch.deg2rad(torch.tensor(ra_deg,  dtype=torch.float32))
    dec_rad = torch.deg2rad(torch.tensor(dec_deg, dtype=torch.float32))

    # 2) compute sin/cos
    ra_cos,  ra_sin  = torch.cos(ra_rad),  torch.sin(ra_rad)
    dec_cos, dec_sin = torch.cos(dec_rad), torch.sin(dec_rad)

    # 3) stack into [cos(RA), sin(RA), cos(DEC), sin(DEC)]
    return torch.stack([ra_cos, ra_sin, dec_cos, dec_sin])



class CustomDataset(Dataset):
    def __init__(self, file_path,  transform=None,new=False):


        self.transform = transform
        self.df = pd.read_csv(file_path)
        self.new_path='/content/processed-imagesp_new/'
        self.image_paths=self.df['image_path'].apply(replace_image_path) 
        #if  new :
        #      self.image_paths= self.df['image_path'] #.apply(join_path)
        self.angle_values = self.df[['raan','dec']]
        self.dec_angle_values = self.df[['dec']]
        self.angles_between_points = self.df['polygon_angles'] # Convert string representation of list to actual list



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.image_paths.iloc[idx]
        if  image.startswith('/'):
           image='/'.join(image.split('/')[1:])
        img=cv2.imread(image)
        #angle = random.uniform(-2, 2)
        #masked_image= torch.tensor(preprocess_image(image)[0]).to(torch.float32)  # Load mask using the provided function
        img=rotate_image_about_center(img,0)
        masked_image= torch.tensor(img).to(torch.float32)
        #print(image)
        
        #print(masked_image.shape)

        #print(masked_image.shape)
        angles = self.angle_values.iloc[idx].tolist()  # Ensure angles are float
        #print(angles)
        angles=torch.tensor(angles)
        angles = torch.fmod(angles+ 360, 360)
        angle_label = encode_labels(angles[0], angles[1]) 
        

        angles_between_points = json.loads(self.angles_between_points.iloc[idx])





        # Convert the list of angles to a PyTorch tensor
        angles_between_points = torch.tensor(angles_between_points, dtype=torch.float32)

        angles_between_points = torch.fmod(angles_between_points + 360, 360)

        # Normalize angles using cosine to get values between 0 and 1
        angles_between_points_rad = torch.deg2rad(angles_between_points)
        enc = torch.stack([torch.sin(angles_between_points_rad), torch.cos(angles_between_points_rad)], dim=1).flatten()  # size = 2*N
        # Create a padded tensor with a size of 8 (if needed)
        padded_tensor = torch.zeros(16)
        padded_tensor[:enc.size(0)] = enc
          # size = 2*N
        # Now you can continue with the rest of your training or inference process





        #print(padded_tensor.shape)

        return masked_image, angle_label,padded_tensor,

# Initialize dataset and dataloader
"""

data_file="merged_data.csv"

data_file2='angles_between_polygons_new.csv'

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
dataset = CustomDataset(data_file)
dataset2 = CustomDataset(data_file,new=True)
#print(len(dataset2))
#print(len(dataset))
#dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

combined_dataset = ConcatDataset([dataset,dataset2])

validation_split_ratio = 0.2
dataset_size = len(dataset)


validation_size = int(validation_split_ratio * dataset_size)
train_size = dataset_size - validation_size
"""
