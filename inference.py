import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import PromptIR


# Paths
ckpt_path = r"C:\Users\user\OneDrive\桌面\cv_hw4\epoch=75-step=243200.ckpt"  # replace with your .ckpt path
test_dir = r"C:\Users\user\OneDrive\桌面\cv_hw4\hw4_realse_dataset\test\degraded"
output_npz = "pred.npz"

# Image transform
to_tensor = transforms.ToTensor()


class InferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)

    def forward(self, x):
        return self.net(x)


# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
model = InferenceModel()
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval().to(device)

# Collect predictions
results = {}

with torch.no_grad():
    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = to_tensor(img).unsqueeze(0).to(device)

        output = model(img_tensor).squeeze(0).clamp(0, 1).cpu().numpy()
        output = (output * 255).astype(np.uint8)

        results[fname] = output

np.savez(output_npz, **results)
print(f"✅ Inference complete. Saved {len(results)} images to {output_npz}")
