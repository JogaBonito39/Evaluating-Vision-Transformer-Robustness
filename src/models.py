import torch
import torchvision.models as models
import timm

def get_resnet18():
    #load ResNet-18 with ImageNet-1K pretained weights
    model = models.resnet18(weights='DEFAULT')
    model.eval()
    return model

def get_vit_tiny():
    #load Vision Tranformer (Tiny verison)
    # model = models.vit_b_16(weights='DEFAULT') # vit_b_16 is standard; vit_tiny often comes from 'timm' library
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.eval()
    return model

def move_to_device(model):
    # Detect if a GPU (CUDA or MPS for Mac) is available, otherwise use CPU
    device = torch.device("cuda" if torch.cude.is_available() else "cpu")
    return model.to(device), device