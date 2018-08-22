# Multi-task Learning For CIFAR-10 Dataset Implemented In PyTorch

## Introduction

Multi-task learning seems sensitive to how it is trained and how its loss function is formed. 
To verify its sensitiveness, several experiments are proposed.

## Usage

### Train

```
python main.py --train
```

Arguments:

 * `--setting`: (default: `0`)
   * `0`: Standard CIFAR-10 experiment. Train a CIFAR-10 multi-class classifier on full dataset.
   * `1`: The training setting is the same as setting `0`, but instead of recording the multiclass classification accuracy, it records the binary classification accuracy for each class. 
 * `--save_path`: Path (directory) that model and history are saved. (default: `'.'`)
 * `--save_model`: A flag used to decide whether to save model or not.
 * `--save_history`: A flag used to decide whether to save training history or not.
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.

### Evaluate

```
python main.py --train
```

Arguments:

 * `--setting`: (default: `0`)
   * `0`: Standard CIFAR-10 experiment. Evaluate a CIFAR-10 multi-class classifier on testing dataset.
   * `1`: The model is the same as setting `0`, but instead of recording the multiclass classification accuracy, it records the binary classification accuracy for each class. 
 * `--save_path`: Path (directory) that model is saved. (default: `'.'`)
