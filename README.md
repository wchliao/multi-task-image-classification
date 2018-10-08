# Multi-task Learning For Image Classification Implemented In PyTorch

## Introduction

Multi-task learning seems sensitive to how it is trained and how its loss function is formed. 
To verify its sensitivity, several experiments are proposed.

## Task Definitions

### CIFAR-10

In a standard task, a classifier is trained to classify images into the 10 classes.  
In a single task *i*, a classifier is trained to distinguish whether images belong to class *i* or not.  
In a multi task, 10 classifiers are trained. For classifier *i*, it is trained to distinguish whether images belong to class *i* or not.

### CIFAR-100

In a standard task, a classifier is trained to classify images into the 100 classes.  
In a single task *i*, a classifier is trained to classify images of coarse *i* into 5 classes.  
In a multi task, 20 classifiers are trained. For classifier *i*, it is trained to classify images of coarse *i* into 5 classes.

### Omniglot

Omniglot is a dataset that contains 1623 different handwritten characters from 50 different alphabets.

In a standard task, a classifier is trained to classify images into the 1623 classes.  
In a single task *i*, a classifier is trained to classify images of alphabet *i*.  
In a multi task, 50 classifiers are trained. For classifier *i*, it is trained to classify images of alphabet *i*.

**Note that the network architecture in this repo is not designed for Omniglot. It is very likely to run out of memory with the architecture.  
To run on Omniglot, please modify the network architecture in `models.py`.**

## Usage

### Train

```
python main.py --train
```

Arguments:

 * `--setting`: (default: `0`)
   * `0`: Train a standard task classifier.
   * `1`: Train a standard task classifier like setting `0`. However, instead of recording the standard task accuracy, accuracies of each single task are recorded.
   * `2`: Train a single task classifier for task *i*.
   * `3`: Train a multi-task model, which contains a classifier for each task. For each iteration, randomly choose a task (in uniform distribution) to train.
   * `4`: Train a multi-task model, which contains a classifier for each task. For each iteration, randomly choose a task (in non-uniform distribution) to train.
   * `5`: Train a multi-task model, which contains a classifier for each task, with a unweighted summed loss. (Only applicable for CIFAR-10)
   * `6`: Train a multi-task model, which contains a classifier for each task, with a weighted summed loss. (Only applicable for CIFAR-10)
 * `--data`: (default: `1`)
   * `0`: CIFAR-10
   * `1`: CIFAR-100
   * `2`: Omniglot
 * `--task`: Task ID (for setting `2`) (default: None) 
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
   * `0`: Evaluate a standard task classifier.
   * `1`: Evaluate a standard task classifier by evaluating each of its single task.
   * `2`: Evaluate a single task classifier for task *i*.
   * `3`: Evaluate a multi-task model for each task.
   * `4`: Same as `3`. 
   * `5`: Evaluate a multi-task model for each task. (Only applicable for CIFAR-10)
   * `6`: Same as `5`. 
 * `--data`: (default: `1`)
   * `0`: CIFAR-10
   * `1`: CIFAR-100
   * `2`: Omniglot
 * `--task`: Task ID (for setting `2`) (default: None)
 * `--save_path`: Path (directory) that model is saved. (default: `'.'`)
