import torch
import json
from src.data_loader import get_loader
from src.models import get_resnet18, get_vit_tiny

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_experiment(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet18":
        model = get_resnet18().to(device)
    else:
        model = get_vit_tiny().to(device)

    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    all_results = {}

    for corr in corruptions:
        print(f"Testing {model_name} on {corr}...")
        severity_accs = []
        for sev in range(1, 6):
            loader = get_loader(root_dir="./data/CIFAR-10-C", corruption=corr, severity=sev)
            acc = evaluate_model(model, loader, device)
            severity_accs.append(acc)

        all_results[corr] = severity_accs

    with open(f"results_{model_name}.json", "w") as f:
        json.dump(all_results, f)

    return all_results