import matplotlib.pyplot as plt

import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, nargs='+')
    parser.add_argument('--name', type=str, nargs='*', default=None)
    parser.add_argument('--title', type=str, default='Training Curve')
    parser.add_argument('--xlabel', type=str, default='Epochs')
    parser.add_argument('--ylabel', type=str, default='Accuracy')
    parser.add_argument('--figure_num', type=int, default=None)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--display', action='store_true')

    parser.add_argument('--filename', type=str, default='figure.png')

    return parser.parse_args()


def plot_figure(y, name=None, title=None, xlabel=None, ylabel=None, figure_num=None, display=True, save=False, filename='figure.png'):
    if figure_num is None:
        plt.figure()
    else:
        plt.figure(figure_num)

    plt.clf()

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if name is None:
        for y_ in y:
            plt.plot(y_)
    else:
        for y_, name_ in zip(y, name):
            plt.plot(y_, label=name_)
        plt.legend(loc='lower right')

    if save:
        plt.savefig(filename)
    if display:
        plt.show()


def main():
    args = parse_args()

    y = []
    for data in args.data:
        with open(data, 'r') as f:
            y.append(json.load(f))

    plot_figure(y=y,
                name=args.name,
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                figure_num=args.figure_num,
                display=args.display,
                save=args.save,
                filename=args.filename
                )


if __name__ == '__main__':
    main()