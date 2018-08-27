import os
import json
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', '-t', type=int, default=1)
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--input2', '-j', type=str)
    parser.add_argument('--num', '-n', type=int, default=1)
    parser.add_argument('--output', '-o', type=str, default='average.json')

    return parser.parse_args()


def average_in_directory(filename, num):
    data = []

    for i in range(num):
        with open(filename + str(i) + '.json', 'r') as f:
            data.append(json.load(f))

    data = [np.mean(d) for d in zip(*data)]

    return data


def average_across_directories(dirname, filename, num):
    data = []

    for i in range(num):
        with open(os.path.join(dirname + str(i), filename) + '.json', 'r') as f:
            data.append(json.load(f))

    data = [np.mean(d) for d in zip(*data)]

    return data


def main():
    args = parse_args()

    if args.type == 1:
        data = average_in_directory(args.input, args.num)
    elif args.type == 2:
        data = average_across_directories(args.input, args.input2, args.num)
    else:
        raise ValueError('Unknown type: {}'.format(args.type))

    with open(args.output, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()