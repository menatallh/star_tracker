import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from models import *
from dataset import *
from torch.utils.data import DataLoader, Dataset

from torch.utils.data import ConcatDataset, WeightedRandomSampler

from torch.utils.data import random_split, DataLoader



class Trainer:
    def __init__(self, model, data_loader,val_loader, criterion, device='cuda', learning_rate=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-4, num_epochs=120, warmup_steps=5000):
        self.model = model
        self.train_loader = data_loader
        self.val_loader=val_loader
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps

        # Initialize the optimizer: Adam with the given parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

        # Calculate total training steps for the cosine schedule with warmup
        total_training_steps = len(data_loader) * num_epochs

        # Initialize the cosine annealing scheduler with linear warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )

    # Function to apply gradient clipping
    def clip_gradients(self, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
    @torch.no_grad()
    def validate(self):
      self.model.eval()
      total_loss = 0.0

      for images, targets, angles in self.val_loader:
          images, targets = images.to(self.device), targets.to(self.device)
          targets=targets.reshape(-1,4)
          targets = targets.to(torch.float32)
          #angles = angles.to(self.device):
          images = images.to(self.device).permute(0,3,1,2)  # if needed
          angle_inputs = angles.to(self.device)
          #targets = targets.to(self.device)
          #print(angles.shape)
          outputs = self.model(images, angle_inputs)
          loss = angular_loss(outputs, targets)
          total_loss += loss.item()

      avg_val_loss = total_loss / len(self.val_loader)
      return avg_val_loss

    def train(self):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for images, targets, angles in self.train_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                targets=targets.reshape(-1,4)
                targets = targets.to(torch.float32)
                #print(targets)                
                angles = angles.to(self.device)
                #print(angles.shape)
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(torch.permute(images, (0, 3, 1, 2)), angles)
                #print(outputs.shape,'output')
                # Calculate the loss
                loss = angular_loss(outputs, targets)
                #loss = self.criterion(outputs, targets)

                # Backward pass
                loss.backward()

                self.clip_gradients()
                self.optimizer.step()
                self.scheduler.step()   # step per batch
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            val_loss = self.validate()

        
            print(f"Epoch {epoch+1}/{self.num_epochs} — Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            # optional: save intermediate snapshots
            if (epoch+1) % 10 == 0:
                 torch.save(self.model.state_dict(), f"VitRegressionPoly_epoch{epoch+1}.pth")

        # finally save the last model
        torch.save(self.model.state_dict(), "VitRegressionPoly_final.pth")
        print("Training complete. Final model saved.")




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Assuming your model class is defined
# Define the model

def angular_loss(preds, labels):
    # preds, labels: [N, 4] (cos(RA), sin(RA), cos(DEC), sin(DEC))
    pred_ra  = torch.atan2(preds[:, 1], preds[:, 0])
    true_ra  = torch.atan2(labels[:, 1], labels[:, 0])
    ra_loss  = torch.mean(1 - torch.cos(pred_ra - true_ra))

    pred_dec = torch.atan2(preds[:, 3], preds[:, 2])
    true_dec = torch.atan2(labels[:, 3], labels[:, 2])
    dec_loss = torch.mean(1 - torch.cos(pred_dec - true_dec))

    return (ra_loss + dec_loss) / 2


def angular_loss(preds, labels):
    # preds, labels: [N, 4] (cos(RA), sin(RA), cos(DEC), sin(DEC))
    pred_ra  = torch.atan2(preds[:, 1], preds[:, 0])
    true_ra  = torch.atan2(labels[:, 1], labels[:, 0])
    ra_loss  = torch.mean(1 - torch.cos(pred_ra - true_ra))

    pred_dec = torch.atan2(preds[:, 3], preds[:, 2])
    true_dec = torch.atan2(labels[:, 3], labels[:, 2])
    dec_loss = torch.mean(1 - torch.cos(pred_dec - true_dec))

    return (ra_loss + dec_loss) / 2

model = ViTForRegressionWithAngles()


criterion = nn.MSELoss()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset and dataloader


data_file="merged_data.csv"
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
stars_dataset = CustomDataset(data_file)


#print(stars_dataset)


data_file2='merged_data_new.csv'


dataset2 = CustomDataset(data_file2,new=True)
#print(len(dataset2))


combined_dataset = ConcatDataset([dataset2,dataset2])

#dataloader = DataLoader(combined_dataset, batch_size=64, shuffle=True)


validation_split_ratio = 0.2
dataset_size = len(combined_dataset)

batch_size=64

validation_size = int(validation_split_ratio * dataset_size)

train_size = dataset_size - validation_size

# Step 2: Split dataset
train_dataset, val_dataset = random_split(combined_dataset, [train_size, validation_size])

# Step 3: Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset and dataloader






# Specify the device (use 'cuda' if available)
#ckpt_path='/content/drive/MyDrive/VitRegressionPoly220'
# Optionally, load from checkpoint if available
# Uncomment and modify the path to load an existing checkpoint
#checkpoint = torch.load(ckpt_path)
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#loss = checkpoint['loss']


#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Restore scheduler state

learning_rate=2e-4

# Initialize the trainer
trainer = Trainer(
    model=model,
    data_loader=train_loader,
    val_loader=val_loader,      # Assuming dataloader is initialized properly
    criterion=criterion,
    device=device,              # 'cuda' or 'cpu'
    learning_rate=learning_rate, # Initial learning rate
    num_epochs=140,       # Total number of epochs
    warmup_steps=5000    # Number of warmup steps (5000 steps)
)

# Start training
trainer.train()
