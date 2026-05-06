import torch.nn as nn
import torch
import torchvision.models as models
import timm
from vit_pytorch import ViT
from torch.hub import load_state_dict_from_url
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import ViTForImageClassification

class ViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Hugging Face models return a dictionary/object; we just want the logits
        return self.model(x).logits
    
def get_resnet18():
    #load ResNet-18 with ImageNet-1K pretained weights
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    # model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    # model = models.resnet18(weights='DEFAULT')
    # nums_ftrs = model.fc.in_features
    # model.fc = nn.Linear(nums_ftrs, 10)
    model.eval()
    return model

def get_vit_tiny():
    #load Vision Tranformer (Tiny verison)
    # model = models.vit_b_16(weights='DEFAULT') # vit_b_16 is standard; vit_tiny often comes from 'timm' library
    # model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10, checkpoint_path='')
    # model = torch.hub.load("edugp/vit-tiny-patch16-224-cifar10", "vit_tiny_patch16_224_cifar10", pretrained=True)
    # model=timm.create_model('hf_hub:winvvw/vit-tiny-patch16-224-cifar10', pretrained=True)
    # model = ViT(
    #     image_size = 32,
    #     patch_size = 4,
    #     num_classes = 10,
    #     dim = 256,      # Tiny dimension
    #     depth = 6,      # Shallow depth for "Tiny"
    #     heads = 8,
    #     mlp_dim = 512,
    #     dropout = 0.1,
    #     emb_dropout = 0.1
    # )

    repo_id = "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
    
    # This automatically handles the 'model.safetensors' and the config
    raw_model = ViTForImageClassification.from_pretrained(repo_id)
    
    model = ViTWrapper(raw_model)
    model.eval()
    return model

def move_to_device(model):
    # Detect if a GPU (CUDA or MPS for Mac) is available, otherwise use CPU
    device = torch.device("cuda" if torch.cude.is_available() else "cpu")
    return model.to(device), device