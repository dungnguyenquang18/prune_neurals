import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from prunner import Prunner
import copy
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import argparse
import time
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 100

# MNIST transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


# ── LeNet 300-100 ────────────────────────────────────────────────────────────
class LeNet300_100(nn.Module):
    def __init__(self):
        super(LeNet300_100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ── Helpers ──────────────────────────────────────────────────────────────────
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
    recall    = recall_score(all_labels, all_predictions, average='weighted')
    f1        = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-Score:  {f1:.4f}')

    return accuracy, precision, recall, f1


def parse_args():
    parser = argparse.ArgumentParser(description="LeNet 300-100 Pruning Script")
    parser.add_argument(
        "-method",
        type=str,
        default="base",
        choices=['base', 'kmeans', 'distance', 'kmedoids'],
        help="Pruning method to use"
    )
    return parser.parse_args()


# Layer accessor helpers so we can address fc1/fc2/fc3 by combo index
LAYER_NAMES = ['fc1', 'fc2', 'fc3']


def get_layer(model, idx):
    return getattr(model, LAYER_NAMES[idx])


def set_layer(model, idx, layer):
    setattr(model, LAYER_NAMES[idx], layer)


def prune(model, combo, prune_ratio, args):
    test_model = copy.deepcopy(model)

    num_workers = multiprocessing.cpu_count()
    print(f"\n{'='*60}")
    print(f"PARALLEL PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Available CPU cores: {num_workers}")
    print(f"{'='*60}\n")

    pruner = Prunner()
    start_time = time.time()

    new_layer_1, new_layer_2 = pruner.prune_neurals(
        get_layer(test_model, combo[0]),
        get_layer(test_model, combo[1]),
        prune_ratio=prune_ratio,
        method=args.method,
        device='cpu'
    )

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Pruning completed in {total_time:.2f} seconds")
    print(f"{'='*60}\n")

    set_layer(test_model, combo[0], new_layer_1)
    set_layer(test_model, combo[1], new_layer_2)

    # Debug: print fc structure after pruning
    print("\nModel structure after pruning:")
    for name in LAYER_NAMES:
        layer = getattr(test_model, name)
        print(f"{name}: in={layer.in_features}, out={layer.out_features}, "
              f"weight={tuple(layer.weight.data.shape)}, bias={tuple(layer.bias.data.shape)}")

    accuracy, precision, recall, f1 = test(test_model)

    return total_time, accuracy, precision, recall, f1


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    results = {}
    print(f"Using pruning method: {args.method}")

    model = LeNet300_100()
    model_path = 'mnist_lenet_300_100.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to('cpu')
    print("Model loaded successfully")

    # Debug: print structure before pruning
    print("\nModel structure before pruning:")
    for name in LAYER_NAMES:
        layer = getattr(model, name)
        print(f"{name}: in={layer.in_features}, out={layer.out_features}")

    # Pairs: (fc1, fc2) and (fc2, fc3)  — indices into LAYER_NAMES
    combo_layers = [(0, 1), (1, 2)]

    for idx, combo in enumerate(combo_layers):
        results[idx] = {}
        for prune_ratio in [0.3, 0.5, 0.7, 0.9]:
            results[idx][prune_ratio] = {}
            total_time = accuracy = precision = recall = f1 = 0

            for trial in range(3):
                t, a, p, r, f = prune(model, combo, prune_ratio, args)
                total_time += t
                accuracy   += a
                precision  += p
                recall     += r
                f1         += f
                print(f"Trial {trial+1}/3 | combo={combo} | ratio={prune_ratio:.1f} | "
                      f"time={t:.2f}s | acc={a:.2f}% | prec={p:.4f} | rec={r:.4f} | f1={f:.4f}")

            results[idx][prune_ratio]['time']      = total_time / 3
            results[idx][prune_ratio]['accuracy']  = accuracy   / 3
            results[idx][prune_ratio]['precision'] = precision  / 3
            results[idx][prune_ratio]['recall']    = recall     / 3
            results[idx][prune_ratio]['f1']        = f1         / 3
            print("Done | combo={} | ratio={:.1f}".format(combo, prune_ratio))

        print(f"\nCompleted layers {combo[0]}-{combo[1]}:")
        for prune_ratio in results[idx]:
            r = results[idx][prune_ratio]
            print(f"  ratio={prune_ratio:.1f} time={r['time']:.2f}s acc={r['accuracy']:.2f}% "
                  f"prec={r['precision']:.4f} rec={r['recall']:.4f} f1={r['f1']:.4f}")

    os.makedirs('results/lenet', exist_ok=True)
    out_path = f'results/lenet/{args.method}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path}")

    print("\nFinal Results:")
    for idx, combo in enumerate(combo_layers):
        layer_a = LAYER_NAMES[combo[0]]
        layer_b = LAYER_NAMES[combo[1]]
        print(f"\nPruning {layer_a} -> {layer_b}:")
        for prune_ratio in results[idx]:
            r = results[idx][prune_ratio]
            print(f"  ratio={prune_ratio:.1f} time={r['time']:.2f}s acc={r['accuracy']:.2f}% "
                  f"prec={r['precision']:.4f} rec={r['recall']:.4f} f1={r['f1']:.4f}")
