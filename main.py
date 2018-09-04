import numpy as np
import argparse
from agents import SingleTaskAgent, StandardAgent, MultiTaskSeparateAgent, MultiTaskJointAgent
from utils import CIFAR10Loader, CIFAR100Loader


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--setting', type=int, default=0, help='0: Standard CIFAR experiment \n'
                                                               '1: Standard CIFAR experiment (recording each class\' accuracy separately) \n'
                                                               '2: Single task experiment \n'
                                                               '3: Multi-task experiment (trained separately) \n'
                                                               '4: Multi-task experiment (trained separately with biased sample probability) \n'
                                                               '5: Multi-task experiment (trained jointly) \n'
                                                               '6: Multi-task experiment (trained jointly with biased weighted loss)')
    parser.add_argument('--task', type=int, default=None, help='Which class to distinguish (for setting 2)')
    parser.add_argument('--CIFAR10', action='store_true')
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_history', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    if not args.CIFAR10 and (args.setting == 5 or args.setting == 6):
        raise ValueError('CIFAR-100 is not applicable to setting 5 and 6.')

    if args.CIFAR10:
        train_data = CIFAR10Loader(batch_size=128, train=True)
        test_data = CIFAR10Loader(batch_size=128, train=False)
        num_classes = 10
        num_tasks = 10
        num_subclasses = 2
        num_epochs = 100
    else:
        train_data = CIFAR100Loader(batch_size=128, train=True)
        test_data = CIFAR100Loader(batch_size=128, train=False)
        num_classes = 100
        num_tasks = 20
        num_subclasses = 5
        num_epochs = 200

    if args.setting == 0:
        agent = SingleTaskAgent(num_classes=num_classes)
        train_data = train_data.get_loader()
        test_data = test_data.get_loader()
    elif args.setting == 1:
        agent = StandardAgent(CIFAR10=args.CIFAR10)
        train_data = train_data.get_loader()
    elif args.setting == 2:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)
        agent = SingleTaskAgent(num_classes=num_subclasses)
        train_data = train_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)
    elif args.setting == 3:
        agent = MultiTaskSeparateAgent(num_tasks=num_tasks, num_classes=num_subclasses)
    elif args.setting == 4:
        prob = np.arange(num_tasks) + 1
        prob = prob / sum(prob)
        agent = MultiTaskSeparateAgent(num_tasks=num_tasks, num_classes=num_subclasses, task_prob=prob.tolist())
    elif args.setting == 5:
        agent = MultiTaskJointAgent(num_tasks=num_tasks, num_classes=num_subclasses)
    elif args.setting == 6:
        weight = np.arange(num_tasks) + 1
        weight = weight / sum(weight)
        agent = MultiTaskJointAgent(num_tasks=num_tasks, num_classes=num_subclasses, loss_weight=weight.tolist())
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    agent.train(train_data=train_data,
                test_data=test_data,
                num_epochs=num_epochs,
                save_history=args.save_history,
                save_path=args.save_path,
                verbose=args.verbose
                )

    if args.save_model:
        agent.save_model(args.save_path)


def eval(args):
    if not args.CIFAR10 and (args.setting == 5 or args.setting == 6):
        raise ValueError('CIFAR-100 is not applicable to setting 5 and 6.')

    if args.CIFAR10:
        data = CIFAR10Loader(batch_size=128, train=False)
        num_classes = 10
        num_tasks = 10
        num_subclasses = 2
    else:
        data = CIFAR100Loader(batch_size=128, train=False)
        num_classes = 100
        num_tasks = 20
        num_subclasses = 5

    if args.setting == 0:
        agent = SingleTaskAgent(num_classes=num_classes)
        data = data.get_loader()
    elif args.setting == 1:
        agent = StandardAgent(CIFAR10=args.CIFAR10)
    elif args.setting == 2:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)
        agent = SingleTaskAgent(num_classes=num_subclasses)
        data = data.get_loader(args.task)
    elif args.setting == 3 or args.setting == 4:
        agent = MultiTaskSeparateAgent(num_tasks=num_tasks, num_classes=num_subclasses)
    elif args.setting == 5 or args.setting == 6:
        agent = MultiTaskJointAgent(num_tasks=num_tasks, num_classes=num_subclasses)
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    agent.load_model(args.save_path)
    accuracy = agent.eval(data)

    print('Accuracy: {}'.format(accuracy))


def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        eval(args)
    else:
        print('No flag is assigned. Please assign either \'--train\' or \'--eval\'.')


if __name__ == '__main__':
    main()