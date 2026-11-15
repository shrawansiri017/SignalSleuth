
import sys
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL = "bendavidsteel/openfake-swinv2-base"

print("Loading OpenFake image detector...")

processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForImageClassification.from_pretrained(MODEL)
model.eval()

for path in sys.argv[1:]:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = (logits / 2.0).softmax(dim=-1)[0]

    real_score = probabilities[0].item() * 100
    fake_score = probabilities[1].item() * 100

    print("\nImage:", path)
    print(f"REAL: {real_score:.2f}%")
    print(f"FAKE: {fake_score:.2f}%")
