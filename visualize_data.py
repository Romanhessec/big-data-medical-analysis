"""
Visualize a specific medical image (x-ray) of .pkl format
"""

import pickle
import matplotlib.pyplot as plt
import torch

# Load the .pkl file
file_path = 'chexlocalize/CheXlocalize/gradcam_maps_val/patient64738_study1_view1_frontal_Consolidation_map.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Extract the chest X-ray tensor ('cxr_img')
cxr_img = data['cxr_img']

# Convert it to a NumPy array
if isinstance(cxr_img, torch.Tensor):
    cxr_img = cxr_img.numpy()

# Plot the image
plt.imshow(cxr_img[0], cmap='gray')
plt.title(f"Task: {data['task']} (Ground Truth: {data['gt']})")
plt.axis('off')

# Save the image to a file
output_path = 'output_image.png'
plt.savefig(output_path)
print(f"Image saved to {output_path}")
