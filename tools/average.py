import json
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--num', '-n', type=int, default=1)
    parser.add_argument('--output', '-o', type=str, default='average.json')

    return parser.parse_args()


def main():
    args = parse_args()

    data = []

    for i in range(args.num):
        with open(args.input + str(i) + '.json', 'r') as f:
            data.append(json.load(f))

    data = [np.mean(d) for d in zip(*data)]

    with open(args.output, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()