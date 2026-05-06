import torch
from src.models import get_resnet18
from src.data_loader import get_loader

def run_sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the model and data
    model = get_resnet18().to(device)
    # We use severity 1 fog just as a sample
    loader = get_loader(root_dir="./data/CIFAR-10-C", corruption='fog', severity=1, batch_size=5)
    
    # 2. Grab exactly one batch of 5 images
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    # 3. Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    print("--- SANITY CHECK RESULTS ---")
    print(f"Actual Labels (CIFAR):    {labels.tolist()}")
    print(f"Predicted IDs (Model):   {predicted.tolist()}")
    
    # Explanation logic
    if any(p > 9 for p in predicted):
        print("\n❌ DIAGNOSIS: Class Mismatch!")
        print("The model predicted IDs greater than 9. It's using ImageNet classes (0-999).")
    else:
        print("\n✅ Predicted IDs are within 0-9. Checking for accuracy...")

if __name__ == "__main__":
    run_sanity_check()