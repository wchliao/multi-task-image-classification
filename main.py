import numpy as np
import argparse
from agents import SingleTaskAgent, StandardAgent, MultiTaskSeparateAgent
from utils import CIFAR10Loader


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--setting', type=int, default=0, help='0: Standard CIFAR-10 experiment \n'
                                                               '1: Standard CIFAR-10 experiment (recording each class\' accuracy separately) \n'
                                                               '2: Single task experiment \n'
                                                               '3: Multi-task experiment (trained separately) \n'
                                                               '4: Multi-task experiment (trained separately with biased sample probability)')
    parser.add_argument('--task', type=int, default=None, help='Which class to distinguish (for setting 2)')
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_history', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    train_data = CIFAR10Loader(batch_size=128, train=True)
    test_data = CIFAR10Loader(batch_size=128, train=False)

    if args.setting == 0:
        agent = SingleTaskAgent(num_classes=10)
        train_data = train_data.get_loader()
        test_data = test_data.get_loader()
    elif args.setting == 1:
        agent = StandardAgent(num_classes=10)
        train_data = train_data.get_loader()
        test_data = test_data.get_loader()
    elif args.setting == 2:
        assert args.task in list(range(10)), 'Unknown task: {}'.format(args.task)
        agent = SingleTaskAgent(num_classes=2)
        train_data = train_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)
    elif args.setting == 3:
        agent = MultiTaskSeparateAgent(num_tasks=10, num_classes=2)
    elif args.setting == 4:
        prob = np.arange(1, 11)
        prob = prob / sum(prob)
        agent = MultiTaskSeparateAgent(num_tasks=10, num_classes=2, task_prob=prob)
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    agent.train(train_data=train_data,
                test_data=test_data,
                save_history=args.save_history,
                save_path=args.save_path,
                verbose=args.verbose
                )

    if args.save_model:
        agent.save_model(args.save_path)


def eval(args):
    data = CIFAR10Loader(batch_size=128, train=False)

    if args.setting == 0:
        agent = SingleTaskAgent(num_classes=10)
        data = data.get_loader()
    elif args.setting == 1:
        agent = StandardAgent(num_classes=10)
        data = data.get_loader()
    elif args.setting == 2:
        assert args.task in list(range(10)), 'Unknown task: {}'.format(args.task)
        agent = SingleTaskAgent(num_classes=2)
        data = data.get_loader(args.task)
    elif args.setting == 3 or args.setting == 4:
        agent = MultiTaskSeparateAgent(num_tasks=10, num_classes=2)
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    agent.load_model(args.save_path)
    accuracy = agent.eval(data)

    print('Accuracy: {}'.format(accuracy))


def main():
    args = parse_args()
    if args.train:
        train(args)
    else:
        eval(args)


if __name__ == '__main__':
    main()