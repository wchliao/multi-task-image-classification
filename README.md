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
   * `0`: Standard CIFAR-10 experiment. Train a multiclass classifier to classify CIFAR-10 images.
   * `1`: The training setting is the same as setting `0`, but instead of recording the multiclass classification accuracy, it records the binary classification accuracy for each class.
   * `2`: Single task experiment. Train a binary classifier to distinguish whether an image belongs to a certain class or not.
   * `3`: Multi-task experiment. Train a multi-task model for each task. For each iteration, (uniform) randomly choose a task to train.
   * `4`: Same as `3`, but use a certain (biased) probability to choose tasks to train.
   * `5`: Multi-task experiment. Train a multi-task model for each task with a (uniformly) summed loss.
 * `--task`: Which class to distinguish (for setting `2`) (default: `0`) 
 * `--save_path`: Path (directory) that model and history are saved. (default: `'.'`)
 * `--save_model`: A flag used to decide whether to save model or not.
 * `--save_history`: A flag used to decide whether to save training history or not.
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.

### Evaluate

```
python main.py --eval
```

Arguments:

 * `--setting`: (default: `0`)
   * `0`: Standard CIFAR-10 experiment. Evaluate a CIFAR-10 multi-class classifier on testing dataset.
   * `1`: The model is the same as setting `0`, but instead of recording the multiclass classification accuracy, it records the binary classification accuracy for each class.
   * `2`: Single task experiment. Evaluate a single task on a certain task.
   * `3`: Multi-task experiment (trained separately). Evaluate a multi-task model for each task.
   * `4`: Same as `3`. 
   * `5`: Multi-task experiment (trained jointly). Evaluate a multi-task model for each task.
 * `--save_path`: Path (directory) that model is saved. (default: `'.'`)
