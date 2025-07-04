# Step 1: Install dependencies
!pip install torch torchvision pillow

# Step 2: Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from google.colab import files

# Step 3: Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 4: Image loader
def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path)
    size = max_size if max(image.size) > max_size else max(image.size)
    
    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Step 5: Upload your content and style images
uploaded = files.upload()
content_img_path = list(uploaded.keys())[0]

uploaded = files.upload()
style_img_path = list(uploaded.keys())[0]

content = load_image(content_img_path).to(device)
style = load_image(style_img_path, shape=content.shape[-2:]).to(device)

# Step 6: Display function
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return image.clip(0, 1)

# Step 7: Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Step 8: Content and style layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content layer
                  '28': 'conv5_1'}
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram 

# Step 9: Extract features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate Gram matrices for style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image
target = content.clone().requires_grad_(True).to(device)

# Weights for style layers
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1
}

content_weight = 1e4  # alpha
style_weight = 1e2    # beta

# Step 10: Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Step 11: Training loop
epochs = 300
for i in range(1, epochs+1):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        _, d, h, w = target_feature.shape
        style_loss += layer_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}, Total loss: {total_loss.item()}")

# Step 12: Show result
final_img = im_convert(target)
plt.imshow(final_img)
plt.axis('off')
plt.title("🎨 Stylized Image")
plt.show()
