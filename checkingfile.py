import torch

model_path = (
    r"D:\Medical_Datasets\ChestMNIST"
    r"\best_chestmnist_model.pth"
)

# This is the trusted checkpoint created by your own training.
checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False,
)

print("Your saved model loaded successfully!")

if isinstance(checkpoint, dict):
    print("Saved items:", list(checkpoint.keys())[:20])