import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
import torch.cuda as cuda
import time
import uuid
from prune_neurals import PruneNeurals
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64  


# Data augmentation và chuẩn hóa cho CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load dữ liệu CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
# Hàm test
def test(model_):
    model_.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    
    # Load mô hình VGG16 với weights pretrained
    model = models.vgg16(weights=False)

    # Thay đổi lớp fully connected cuối để phù hợp với CIFAR-10 (10 classes)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10)
    model.to('cpu')
    model.load_state_dict(torch.load("D:/lab/prune_neurals/best_model_cpu.pth"))

    print("load successfully")
    
    # Debug: Print classifier structure before pruning
    print("Classifier structure before pruning:")
    for i, layer in enumerate(model.classifier):
        if hasattr(layer, 'weight'):
            print(f"Layer {i}: {type(layer).__name__} - in_features: {layer.in_features}, out_features: {layer.out_features}")
        else:
            print(f"Layer {i}: {type(layer).__name__}")
    
    test_model = copy.deepcopy(model) 
    pruner = PruneNeurals()
    new_layer_1, new_layer_2 = pruner.prune(test_model.classifier[0], test_model.classifier[3], prune_ratio=0.9, method='kmeans', device='cpu')
    test_model.classifier[0] = new_layer_1
    test_model.classifier[3] = new_layer_2

    # Debug: Print classifier structure after pruning
    print("\nClassifier structure after pruning:")
    for i, layer in enumerate(test_model.classifier):
        if hasattr(layer, 'weight'):
            print(f"Layer {i}: {type(layer).__name__} - in_features: {layer.in_features}, out_features: {layer.out_features}, weight_shape: {layer.weight.data.shape}, bias_shape: {layer.bias.data.shape}")
        else:
            print(f"Layer {i}: {type(layer).__name__}")

    test(test_model)