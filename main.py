import argparse
from agents import StandardAgent, StandardAgentSeparateRecord
from utils import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--setting', type=int, default=0, help='0: Standard CIFAR-10 experiment \n'
                                                               '1: Standard CIFAR-10 experiment (recording each class\' accuracy separately)')
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_history', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    if args.setting == 0:
        agent = StandardAgent()
    elif args.setting == 1:
        agent = StandardAgentSeparateRecord()
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    train_data = DataLoader(batch_size=128, train=True)
    test_data = DataLoader(batch_size=128, train=False)

    agent.train(train_data=train_data,
                test_data=test_data,
                save_history=args.save_history,
                save_path=args.save_path,
                verbose=args.verbose
                )

    if args.save_model:
        agent.save_model(args.save_path)


def eval(args):
    if args.setting == 0:
        agent = StandardAgent()
    elif args.setting == 1:
        agent = StandardAgentSeparateRecord()
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    agent.load_model(args.save_path)
    data = DataLoader(batch_size=128, train=False)
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