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
from prunner import Prunner
import copy
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import argparse


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
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform_test)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Network Pruning Script")

    parser.add_argument(
        "-method",
        type=str,
        default="base",
        choices=['base', 'kmeans', 'distance-based-clustering', 'kmedoids'],
        help="Pruning method to use"
    )

    return parser.parse_args()

def prune(model, combo, prune_ratio, args):
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
        test_model.classifier[combo[0]], 
        test_model.classifier[combo[1]], 
        prune_ratio=prune_ratio, 
        method=args.method, 
        device='cpu',
        max_workers=2  # Sử dụng đa luồng
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Pruning completed in {total_time:.2f} seconds")
    print(f"{'='*60}\n")
    
    test_model.classifier[combo[0]] = new_layer_1
    test_model.classifier[combo[1]] = new_layer_2

    # Debug: Print classifier structure after pruning
    print("\nClassifier structure after pruning:")
    for i, layer in enumerate(test_model.classifier):
        if hasattr(layer, 'weight'):
            print(f"Layer {i}: {type(layer).__name__} - in_features: {layer.in_features}, out_features: {layer.out_features}, weight_shape: {layer.weight.data.shape}, bias_shape: {layer.bias.data.shape}")
        else:
            print(f"Layer {i}: {type(layer).__name__}")

    accuracy, precision, recall, f1 = test(test_model)
    
    return total_time, accuracy, precision, recall, f1

if __name__ == '__main__':
    args = parse_args()
    results = {}
    print(f"Using pruning method: {args.method}")
    model = models.vgg16(weights=False)
    # Thay đổi lớp fully connected cuối để phù hợp với CIFAR-10 (10 classes)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10)
    model.to('cpu')
    model.load_state_dict(torch.load("/home/dev/dungnq57work/pycharm_dug/best_model_vgg16_cpu.pth"))

    print("load successfully")
    
    # Debug: Print classifier structure before pruning
    print("Classifier structure before pruning:")
    for i, layer in enumerate(model.classifier):
        if hasattr(layer, 'weight'):
            print(f"Layer {i}: {type(layer).__name__} - in_features: {layer.in_features}, out_features: {layer.out_features}")
        else:
            print(f"Layer {i}: {type(layer).__name__}")
            
            
    combo_layers = [(0, 3), (3, 6)]  # Các cặp lớp cần prune
    for idx,combo in enumerate(combo_layers):
        results[idx] = {}
        for prune_ratio in [0.7, 0.8, 0.9]:# Có thể thêm nhiều prune_ratio khác nếu muốn


            results[idx][prune_ratio] = {}
            total_time, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
            for trial in range(3):
                total_time_trial, accuracy_trial, precision_trial, recall_trial, f1_trial = prune(model, combo, prune_ratio, args)
                total_time += total_time_trial
                accuracy += accuracy_trial  
                precision += precision_trial
                recall += recall_trial
                f1 += f1_trial
                # break
                print(f"Trial {trial+1}/3 for combo {combo} with prune ratio {prune_ratio:.1f} completed. Time: {total_time_trial:.2f}s, Accuracy: {accuracy_trial:.2f}%, Precision: {precision_trial:.4f}, Recall: {recall_trial:.4f}, F1-Score: {f1_trial:.4f}")
            results[idx][prune_ratio]['time'] = total_time / 3
            results[idx][prune_ratio]['accuracy'] = accuracy / 3
            results[idx][prune_ratio]['precision'] = precision / 3
            results[idx][prune_ratio]['recall'] = recall / 3
            results[idx][prune_ratio]['f1'] = f1 / 3
            # break
            print("Done pruning trial for combo {} with prune ratio {:.1f}".format(combo, prune_ratio))
        print(f"Completed pruning layers {combo[0]} and {combo[1]} with prune ratio {prune_ratio:.1f}. Time: {results[idx][prune_ratio]['time']:.2f}s, Accuracy: {results[idx][prune_ratio]['accuracy']:.2f}%, Precision: {results[idx][prune_ratio]['precision']:.4f}, Recall: {results[idx][prune_ratio]['recall']:.4f}, F1-Score: {results[idx][prune_ratio]['f1']:.4f}")
        # break
    with open(f'results/{args.method}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to results/{args.method}.json")
    print("\nFinal Results:")
    for idx, combo in enumerate(combo_layers):
        print(f"\nPruning layers {combo[0]} and {combo[1]}:")
        for prune_ratio in results[idx]:
            print(f"  Prune Ratio: {prune_ratio:.1f} - Time: {results[idx][prune_ratio]['time']:.2f}s, Accuracy: {results[idx][prune_ratio]['accuracy']:.2f}%, Precision: {results[idx][prune_ratio]['precision']:.4f}, Recall: {results[idx][prune_ratio]['recall']:.4f}, F1-Score: {results[idx][prune_ratio]['f1']:.4f}")
    