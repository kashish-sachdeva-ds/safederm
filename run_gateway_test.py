import sys
from pathlib import Path
import urllib.request
import os
import random
import torch
from PIL import Image

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
from src.config import IMAGES_DIR
from src.gateway import build_gateway_model, PROMPTS

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BiomedCLIP on {device}...")
    model, preprocess, tokenizer = build_gateway_model(device)
    
    # Get 20 real skin images
    all_skin_images = list(IMAGES_DIR.glob("*.jpg"))
    skin_sample_paths = random.sample(all_skin_images, min(20, len(all_skin_images)))

    # Download 20 random non-skin images
    non_skin_dir = PROJECT_ROOT / "data" / "raw" / "non_skin_samples"
    non_skin_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400", # dog
        "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400", # car
        "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=400", # nature
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400", # cat
        "https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=400", # food
        "https://images.unsplash.com/photo-1526512340740-9217d0159da9?w=400", # architecture
        "https://images.unsplash.com/photo-1510127034890-ba27508e9f1c?w=400", # camera
        "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400", # watch
        "https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=400", # sunglasses
        "https://images.unsplash.com/photo-1550258987-190a2d41a8ba?w=400", # pineapple
    ]
    urls = urls + urls
    
    non_skin_paths = []
    print("Downloading non-skin images...")
    for i, url in enumerate(urls):
        path = non_skin_dir / f"nonskin_{i}.jpg"
        if not path.exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
                out_file.write(response.read())
        non_skin_paths.append(path)

    def get_skin_score(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            text_tokens = tokenizer(PROMPTS).to(device)
            with torch.no_grad():
                image_features, text_features, logit_scale = model(image_tensor, text_tokens)
                logits = (logit_scale * image_features @ text_features.T)
                probs = logits.softmax(dim=-1)
            return probs[0][0].item()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return -1

    print("Evaluating skin images...")
    skin_scores = [get_skin_score(p) for p in skin_sample_paths]
    
    print("Evaluating non-skin images...")
    non_skin_scores = [get_skin_score(p) for p in non_skin_paths]

    print("\n--- RESULTS ---")
    print(f"Average Skin Score: {sum(s for s in skin_scores if s >= 0)/len(skin_scores):.4f}")
    print(f"Min Skin Score: {min(s for s in skin_scores if s >= 0):.4f}")
    print(f"Average Non-Skin Score: {sum(s for s in non_skin_scores if s >= 0)/len(non_skin_scores):.4f}")
    print(f"Max Non-Skin Score: {max(s for s in non_skin_scores if s >= 0):.4f}")

if __name__ == "__main__":
    main()
