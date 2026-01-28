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
from .prunner import Prunner
import copy
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score


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
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    
    # Load mô hình VGG16 với weights pretrained
    model = models.vgg16(weights=False)

    # Thay đổi lớp fully connected cuối để phù hợp với CIFAR-10 (10 classes)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10)
    model.to('cpu')
    model.load_state_dict(torch.load("D:/lab/prune_neurals/best_model_vgg16_cpu.pth"))

    print("load successfully")
    
    # Debug: Print classifier structure before pruning
    print("Classifier structure before pruning:")
    for i, layer in enumerate(model.classifier):
        if hasattr(layer, 'weight'):
            print(f"Layer {i}: {type(layer).__name__} - in_features: {layer.in_features}, out_features: {layer.out_features}")
        else:
            print(f"Layer {i}: {type(layer).__name__}")
    
    test_model = copy.deepcopy(model)
    
    # Cấu hình đa luồng
    num_workers = multiprocessing.cpu_count()
    print(f"\n{'='*60}")
    print(f"PARALLEL PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Available CPU cores: {num_workers}")
    print(f"Using {num_workers} worker threads for parallel cluster processing")
    print(f"{'='*60}\n")
    
    pruner = Prunner()
    start_time = time.time()
    
    # Gọi prune_neurals với max_workers để sử dụng đa luồng
    new_layer_1, new_layer_2 = pruner.prune_neurals(
        test_model.classifier[3], 
        test_model.classifier[6], 
        prune_ratio=0.6, 
        method='base', 
        device='cpu',
        max_workers=2  # Sử dụng đa luồng
    )
    
    end_time = time.time()
    print(f"\n{'='*60}")
    print(f"Pruning completed in {end_time - start_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    test_model.classifier[3] = new_layer_1
    test_model.classifier[6] = new_layer_2

    # Debug: Print classifier structure after pruning
    print("\nClassifier structure after pruning:")
    for i, layer in enumerate(test_model.classifier):
        if hasattr(layer, 'weight'):
            print(f"Layer {i}: {type(layer).__name__} - in_features: {layer.in_features}, out_features: {layer.out_features}, weight_shape: {layer.weight.data.shape}, bias_shape: {layer.bias.data.shape}")
        else:
            print(f"Layer {i}: {type(layer).__name__}")

    test(model)