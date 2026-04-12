import torch
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from models import ViTForRegressionWithAngles  # your model class
from dataset import *  # replace with your dataset class
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import time
# ---------------------------------------
# 1. Utility functions
# ---------------------------------------

@torch.no_grad()
def angular_loss(preds, labels):
    # preds, labels: [N, 4] (cos(RA), sin(RA), cos(DEC), sin(DEC))
    pred_ra  = torch.atan2(preds[:, 1], preds[:, 0])
    true_ra  = torch.atan2(labels[:, 1], labels[:, 0])
    ra_loss  = torch.mean(1 - torch.cos(pred_ra - true_ra))

    pred_dec = torch.atan2(preds[:, 3], preds[:, 2])
    true_dec = torch.atan2(labels[:, 3], labels[:, 2])
    dec_loss = torch.mean(1 - torch.cos(pred_dec - true_dec))

    return (ra_loss + dec_loss) / 2

def encode_labels(ra_deg, dec_deg):
    ra_rad  = torch.deg2rad(torch.tensor(ra_deg,  dtype=torch.float32))
    dec_rad = torch.deg2rad(torch.tensor(dec_deg, dtype=torch.float32))
    ra_cos,  ra_sin  = torch.cos(ra_rad),  torch.sin(ra_rad)
    dec_cos, dec_sin = torch.cos(dec_rad), torch.sin(dec_rad)
    return torch.stack([ra_cos, ra_sin, dec_cos, dec_sin])

# ---------------------------------------
# 2. Load model and weights
# ---------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViTForRegressionWithAngles()
checkpoint = torch.load("VitRegressionPoly_epoch120.pth", map_location=device)
#print(checkpoint.keys())
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# ---------------------------------------
# 3. Prepare validation dataset
# ---------------------------------------
data_file="merged_data.csv"
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
stars_dataset = CustomDataset(data_file)



dataloader = DataLoader(stars_dataset, batch_size=64, shuffle=True)


validation_split_ratio = 0.2



def decode_cos_sin_to_angle_deg(cos_sin):
    """
    cos_sin: Tensor of shape (B, 2), i.e., [cos, sin] pairs
    Returns: Tensor of angles in degrees, shape (B,)
    """
    rad = torch.atan2(cos_sin[:, 1], cos_sin[:, 0])
    deg = torch.rad2deg(rad)
    return (deg) % 360  # Normalize to [0, 360)
import torch

def calculate_accuracy(pred_ra, true_ra, pred_dec, true_dec, tol_deg=2.0):
    # Absolute angular errors
    ra_error = torch.abs(pred_ra - true_ra)
    dec_error = torch.abs(pred_dec - true_dec)

    # Correct prediction if BOTH RA and DEC are within tolerance
    correct = (ra_error <= tol_deg) & (dec_error <= tol_deg)

    accuracy = correct.float().mean().item() * 100.0
    return accuracy



dataset_size = len(stars_dataset)

batch_size=64

validation_size = int(validation_split_ratio * dataset_size)

train_size = dataset_size - validation_size

# Step 2: Split dataset
train_dataset, val_dataset = random_split(stars_dataset, [train_size, validation_size])

# Step 3: Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Replace with your dataset class and preprocessing logic

total_loss = 0
all_preds = []
all_labels = []

with torch.no_grad():

    all_true_ra, all_true_dec = [], []
    all_pred_ra, all_pred_dec = [], []

    for i,batch in enumerate(train_loader):
        images, labels_deg, angles = batch  # labels_deg: [RA, DEC] in degrees

        # Encode labels as sin/cos
        encoded_labels = torch.stack([
            encode_labels(ra, dec) for ra, dec in zip(labels_deg[:, 0], labels_deg[:, 1])
        ])

        images = images.permute(0, 3, 1, 2).to(device)  # (B, C, H, W)
        angles = angles.to(device)
        encoded_labels = encoded_labels.to(device)

        preds = model(images, angles)

        loss = angular_loss(preds, encoded_labels)
        total_loss += loss.item()
        # Decode predicted and true vectors to angles
        #pred_ra_deg  = decode_vector_to_angle(preds[0:2])
        #pred_dec_deg = decode_vector_to_angle(preds[2:4])

        #true_ra_deg  = float(label_deg[0])
        #true_dec_deg = float(label_deg[1])

        #print(f"[{i}]")
        #print(f"🟩 True RA:  {true_ra_deg:.2f}°\tPredicted RA:  {pred_ra_deg:.2f}°")
        #print(f"🟩 True DEC: {true_dec_deg:.2f}°\tPredicted DEC: {pred_dec_deg:.2f}°\n")

        

        all_preds.append(preds.cpu())
        all_labels.append(encoded_labels.cpu())

        all_true_ra.extend(labels_deg[:, 0].cpu().tolist())
        all_true_dec.extend(labels_deg[:, 1].cpu().tolist())
        #all_pred_ra.extend(pred_ra_deg.tolist())
        #all_pred_dec.extend(pred_dec_deg.tolist())


avg_loss = total_loss / len(val_loader)
print(f"\n✅ Average Angular Loss on Validation Set: {avg_loss:.6f}")



def evaluate_model(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()

    all_pred_ra, all_true_ra = [], []
    all_pred_dec, all_true_dec = [], []

    with torch.no_grad():
        for images, labels, angles in dataloader:
            images = images.permute(0, 3, 1, 2).to(device)
            angles = angles.to(device)
            labels = labels.to(device)  # shape: (B, 4)

            # Forward pass
            outputs = model(images, angles)  # shape: (B, 4)

            # Decode RA and DEC
            pred_ra_deg  = decode_cos_sin_to_angle_deg(outputs[:, :2].cpu())
            true_ra_deg  = decode_cos_sin_to_angle_deg(labels[:, :2].cpu())

            pred_dec_deg = decode_cos_sin_to_angle_deg(outputs[:, 2:].cpu())
            true_dec_deg = decode_cos_sin_to_angle_deg(labels[:, 2:].cpu())

            # Collect for later stats or plotting
            all_pred_ra.append(pred_ra_deg)
            all_true_ra.append(true_ra_deg)
            all_pred_dec.append(pred_dec_deg)
            all_true_dec.append(true_dec_deg)

            # Print current batch predictions vs true
            for pra, tra, pdec, tdec in zip(pred_ra_deg, true_ra_deg, pred_dec_deg, true_dec_deg):
                print(f"RA: predicted = {pra:.2f}°, true = {tra:.2f}° | DEC: predicted = {pdec:.2f}°, true = {tdec:.2f}°")

   # Concatenate all batches
    pred_ra = torch.cat(all_pred_ra)
    true_ra = torch.cat(all_true_ra)
    pred_dec = torch.cat(all_pred_dec)
    true_dec = torch.cat(all_true_dec)

    # Error calculations
    ra_error = torch.abs(pred_ra - true_ra)
    dec_error = torch.abs(pred_dec - true_dec)

    ra_mae = ra_error.mean().item()
    dec_mae = dec_error.mean().item()

    ra_mse = torch.mean((pred_ra - true_ra) ** 2).item()
    dec_mse = torch.mean((pred_dec - true_dec) ** 2).item()

    ra_std = ra_error.std().item()
    dec_std = dec_error.std().item()

    print("\n===== Final Evaluation Results =====")
    print(f"RA  MAE  (deg): {ra_mae:.4f}")
    print(f"DEC MAE  (deg): {dec_mae:.4f}")
    print(f"RA  MSE  (deg²): {ra_mse:.4f}")
    print(f"DEC MSE  (deg²): {dec_mse:.4f}")
    print(f"RA  STD  (deg): {ra_std:.4f}")
    print(f"DEC STD (deg): {dec_std:.4f}")

    return {
        "ra_mae": ra_mae,
        "dec_mae": dec_mae,
        "ra_mse": ra_mse,
        "dec_mse": dec_mse,
        "ra_std": ra_std,
        "dec_std": dec_std,
        "pred_ra": pred_ra,
        "true_ra": true_ra,
        "pred_dec": pred_dec,
        "true_dec": true_dec,
    }




    # Optionally return all results for error analysis
#    return torch.cat(all_pred_ra), torch.cat(all_true_ra), torch.cat(all_pred_dec), torch.cat(all_true_dec)

s=time.time()
results=evaluate_model(model, train_loader, device='cpu')
print(time.time()-s)

all_pred_ra=results['pred_ra']
all_pred_dec=results['pred_dec']
all_true_dec=results['true_dec']

all_true_ra=results['true_ra']

accuracy_1deg = calculate_accuracy(
    all_pred_ra, all_true_ra,
    all_pred_dec, all_true_dec,
    tol_deg=1.0
)

accuracy_2deg = calculate_accuracy(
    all_pred_ra, all_true_ra,
    all_pred_dec, all_true_dec,
    tol_deg=2.0
)

accuracy_5deg = calculate_accuracy(
    all_pred_ra, all_true_ra,
    all_pred_dec, all_true_dec,
    tol_deg=5.0
)

print(f"Accuracy @1°  : {accuracy_1deg:.2f}%")
print(f"Accuracy @2°  : {accuracy_2deg:.2f}%")
print(f"Accuracy @5°  : {accuracy_5deg:.2f}%")
