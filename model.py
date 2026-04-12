import torch
import torch.nn as nn
from transformers import ViTModel

class ViTForRegressionWithAngles(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', num_outputs=2, angle_dim=8):
        super(ViTForRegressionWithAngles, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.angle_embedding = nn.Linear(angle_dim, self.vit.config.hidden_size)
        self.regression_head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_outputs)
        )

    def forward(self, pixel_values, angles):
        # Get ViT output
        outputs = self.vit(pixel_values=pixel_values, interpolate_pos_encoding=True)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Process angles and integrate them into the model
        angle_embeddings = self.angle_embedding(angles)
        combined_output = cls_output + angle_embeddings

        # Pass the combined features through the regression head
        regression_output = self.regression_head(combined_output)
        return regression_output

# Instantiate the model
model = ViTForRegressionWithAngles()

# Sample input
images = torch.randn(1, 3, 224, 224)
angles = torch.randn(1, 8)  # Assuming 10-dimensional angle features

# Forward pass
output = model(images, angles)
print(output)
