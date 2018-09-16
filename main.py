import numpy as np
import argparse
from agents import SingleTaskAgent, StandardAgent, MultiTaskSeparateAgent, MultiTaskJointAgent
from utils import CIFAR10Loader, CIFAR100Loader, OmniglotLoader


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--setting', type=int, default=0, help='0: Standard experiment \n'
                                                               '1: Standard experiment (recording each class\' accuracy separately) \n'
                                                               '2: Single task experiment \n'
                                                               '3: Multi-task experiment (trained separately) \n'
                                                               '4: Multi-task experiment (trained separately with biased sample probability) \n'
                                                               '5: Multi-task experiment (trained jointly) \n'
                                                               '6: Multi-task experiment (trained jointly with biased weighted loss)')
    parser.add_argument('--data', type=int, default=1, help='0: CIFAR-10 \n'
                                                            '1: CIFAR-100 \n'
                                                            '2: Omniglot \n')
    parser.add_argument('--task', type=int, default=None, help='Which class to distinguish (for setting 2)')
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_history', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    if args.data == 0:
        train_data = CIFAR10Loader(batch_size=128, train=True, drop_last=True)
        test_data = CIFAR10Loader(batch_size=128, train=False, drop_last=False)
        multi_task_type = 'binary'
        num_epochs = 20
    elif args.data == 1:
        train_data = CIFAR100Loader(batch_size=128, train=True, drop_last=True)
        test_data = CIFAR100Loader(batch_size=128, train=False, drop_last=False)
        multi_task_type = 'multiclass'
        num_epochs = 20
    elif args.data == 2:
        train_data = OmniglotLoader(batch_size=128, train=True, drop_last=True)
        test_data = OmniglotLoader(batch_size=128, train=False, drop_last=False)
        multi_task_type = 'multiclass'
        num_epochs = 100 # Need more tests to determine
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_classes_single = train_data.num_classes_single
    num_classes_multi = train_data.num_classes_multi
    num_tasks = len(num_classes_multi)
    num_channels = train_data.num_channels

    if args.setting == 0:
        agent = SingleTaskAgent(num_classes=num_classes_single,
                                num_channels=num_channels)
        train_data = train_data.get_loader()
        test_data = test_data.get_loader()
    elif args.setting == 1:
        agent = StandardAgent(num_classes_single=num_classes_single,
                              num_classes_multi=num_classes_multi,
                              multi_task_type=multi_task_type,
                              num_channels=num_channels)
        train_data = train_data.get_loader()
    elif args.setting == 2:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)
        agent = SingleTaskAgent(num_classes=num_classes_multi[args.task],
                                num_channels=num_channels)
        train_data = train_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)
    elif args.setting == 3:
        agent = MultiTaskSeparateAgent(num_classes=num_classes_multi,
                                       num_channels=num_channels)
    elif args.setting == 4:
        prob = np.arange(num_tasks) + 1
        prob = prob / sum(prob)
        agent = MultiTaskSeparateAgent(num_classes=num_classes_multi,
                                       num_channels=num_channels,
                                       task_prob=prob.tolist())
    elif args.setting == 5:
        agent = MultiTaskJointAgent(num_classes=num_classes_multi,
                                    multi_task_type=multi_task_type,
                                    num_channels=num_channels)
    elif args.setting == 6:
        weight = np.arange(num_tasks) + 1
        weight = weight / sum(weight)
        agent = MultiTaskJointAgent(num_classes=num_classes_multi,
                                    multi_task_type=multi_task_type,
                                    num_channels=num_channels,
                                    loss_weight=weight.tolist())
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
    if args.data == 0:
        data = CIFAR10Loader(batch_size=128, train=False, drop_last=False)
        multi_task_type = 'binary'
    elif args.data == 1:
        data = CIFAR100Loader(batch_size=128, train=False, drop_last=False)
        multi_task_type = 'multiclass'
    elif args.data == 2:
        data = OmniglotLoader(batch_size=128, train=False, drop_last=False)
        multi_task_type = 'multiclass'
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_classes_single = data.num_classes_single
    num_classes_multi = data.num_classes_multi
    num_tasks = len(num_classes_multi)
    num_channels = data.num_channels

    if args.setting == 0:
        agent = SingleTaskAgent(num_classes=num_classes_single,
                                num_channels=num_channels)
        data = data.get_loader()
    elif args.setting == 1:
        agent = StandardAgent(num_classes_single=num_classes_single,
                              num_classes_multi=num_classes_multi,
                              multi_task_type=multi_task_type,
                              num_channels=num_channels)
    elif args.setting == 2:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)
        agent = SingleTaskAgent(num_classes=num_classes_multi[args.task],
                                num_channels=num_channels)
        data = data.get_loader(args.task)
    elif args.setting == 3 or args.setting == 4:
        agent = MultiTaskSeparateAgent(num_classes=num_classes_multi,
                                       num_channels=num_channels)
    elif args.setting == 5 or args.setting == 6:
        agent = MultiTaskJointAgent(num_classes=num_classes_multi,
                                    multi_task_type=multi_task_type,
                                    num_channels=num_channels)
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