import clip
import torch
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Extract text features
with open("image_descriptions_meta_without_number_space.txt", "r") as file:
    lines = file.readlines()

text_features = []
for line in lines:
    text = clip.tokenize([line]).to(device)
    with torch.no_grad():
        text_features.append(model.encode_text(text).cpu().numpy())

# Save text features to a numpy array
text_feature_array = np.array(text_features)
np.save('things1854image_caption_feas_CLIP_ViT_L14.npy', np.squeeze(text_feature_array))

# Extract image features
image_features = []
for i in range(1854):
    image_path = f"analysis_and_figure_drawing/data/THINGS_visual_stimuli_1854/image_{i+1}_ori.jpg"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features.append(model.encode_image(image).cpu().numpy())

# Save image features as a numpy array
image_feature_array = np.array(image_features)
np.save('things1854image_feas_CLIP_ViT_L14.npy',np.squeeze(image_feature_array))


# Extract image features
image_features = []
for i in range(44):
    image_path = f"analysis_and_figure_drawing/data/NSD_test_stimuli_44/image_{i}.png"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features.append(model.encode_image(image).cpu().numpy())
        
image_feature_array = np.array(image_features)
np.save('44image_feas_CLIP_ViT_L14.npy', np.squeeze(image_feature_array))

# Extract text features
with open("analysis_and_figure_drawing/data/NSD_test_stimuli_44/captions_44.txt", "r") as file:
    lines = file.readlines()

text_features = []
for line in lines:
    text = clip.tokenize([line]).to(device)
    with torch.no_grad():
        text_features.append(model.encode_text(text).cpu().numpy())

text_feature_array = np.array(text_features)
np.save('44image_caption_feas_CLIP_ViT_L14.npy', np.squeeze(text_feature_array))
