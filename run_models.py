
# -------- Run Models with Hugging Face Transformers --------

from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
import requests

model_name= 'Marqo/marqo-ecommerce-embeddings-L'
# model_name = 'Marqo/marqo-ecommerce-embeddings-B'

model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

img = Image.open(requests.get('https://raw.githubusercontent.com/marqo-ai/marqo-ecommerce-embeddings/refs/heads/main/images/dining-chairs.png', stream=True).raw).convert("RGB")
image = [img]
text = ["dining chairs", "a laptop", "toothbrushes"]
processed = processor(text=text, images=image, padding='max_length', return_tensors="pt")
processor.image_processor.do_rescale = False
with torch.no_grad():
    image_features = model.get_image_features(processed['pixel_values'], normalize=True)
    text_features = model.get_text_features(processed['input_ids'], normalize=True)

    text_probs = (100 * image_features @ text_features.T).softmax(dim=-1)
    
print(text_probs)
# [1.0000e+00, 8.3131e-12, 5.2173e-12]


# -------- Run Models with OpenCLIP --------

from PIL import Image
import open_clip
import requests
import torch

# Specify model from Hugging Face Hub
model_name = 'hf-hub:Marqo/marqo-ecommerce-embeddings-L'
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

# Preprocess the image and tokenize text inputs
# Load an example image from a URL
img = Image.open(requests.get('https://raw.githubusercontent.com/marqo-ai/marqo-ecommerce-embeddings/refs/heads/main/images/dining-chairs.png', stream=True).raw)
image = preprocess_val(img).unsqueeze(0)
text = tokenizer(["dining chairs", "a laptop", "toothbrushes"])

# Perform inference
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image, normalize=True)
    text_features = model.encode_text(text, normalize=True)

    # Calculate similarity probabilities
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Display the label probabilities
print("Label probs:", text_probs)
# [1.0000e+00, 8.3131e-12, 5.2173e-12]
