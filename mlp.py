import gzip
import sys
import argparse
import numpy as np
import urllib.request
import datetime
import matplotlib.pyplot as plt

from solver import Solver
from model import Model
from config import *


def download_mnist():
    '''Download mnist into the relative path dataset of current dir'''
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    for file in mnist_gz.values():
        file_path = os.path.join(dataset_dir, file)
        if not os.path.exists(file_path):
            print(f'Downloading {file} to {dataset_dir}')
            urllib.request.urlretrieve(mnist_url_base + file, file_path)
            print('Download successfully!')


def load_mnist():
    '''Read mnist .gz file to dict mnist_data'''
    mnist_data = {}
    offset = [16, 8, 16, 8]
    offset_idx = 0
    for key, value in mnist_gz.items():
        file_path = os.path.join(dataset_dir, value)
        with gzip.open(file_path, 'rb') as gz_file:
            mnist_data[key] = np.frombuffer(gz_file.read(), np.uint8, offset=offset[offset_idx])
            offset_idx += 1
    mnist_data['X_train'] = mnist_data['X_train'].reshape(-1, img_size).T
    mnist_data['X_test'] = mnist_data['X_test'].reshape(-1, img_size).T

    for key, value in mnist_data.items():
        print(f'{key}: {value.shape}')
    return mnist_data


def standardize(data):
    avg = np.average(data['X_train'])
    std = np.std(data['X_train'])
    data['X_train'] = (data['X_train'] - avg) / std
    avg = np.average(data['X_test'])
    std = np.std(data['X_test'])
    data['X_test'] = (data['X_test'] - avg) / std
    return data


def predict_with_pretrained(solver, data, mean, min_noise, max_noise):
    intens = []
    accs = []
    num_test = data['X_test'].shape[1]
    batch_mask = np.random.choice(num_test, 25)
    fig = plt.figure()
    first = True
    for intensity in range(min_noise, max_noise + 1):
        if intensity % 10 == 0:
            print(f'predict_with_pretrained(): current intensity is {intensity}')
        X_test = data['X_test'].copy().astype(np.float64)
        Y_test = data['Y_test'].copy()
        noise = np.random.normal(loc=mean, scale=intensity, size=X_test.shape)

        X_test += noise
        avg = np.average(X_test)
        std = np.std(X_test)
        X_test = (X_test - avg) / std
        acc, Y_pred = solver.check_accuracy(X_test, Y_test, 10000)
        intens.append(intensity)
        accs.append(acc)

        axes = []
        for i in range(25):
            b = X_test[:, batch_mask[i]].reshape(28, 28)
            axes.append(fig.add_subplot(5, 5, i + 1))
            subplot_title = f"Label {Y_test[batch_mask[i]]}, Pred {Y_pred[batch_mask[i]]}"
            # axes[-1].set_title(subplot_title)
            plt.xlabel(subplot_title)
            plt.imshow(b)
        fig.suptitle(f'Intensity{intensity} Acc={acc}')
        if first:
            fig.tight_layout(pad=0.3)
            first = False
        plt.savefig(os.path.join(images_dir, f'Intensity_{intensity}.png'))
        # plt.show()
        plt.clf()
    fig = plt.figure()
    plt.plot(intens, accs)
    plt.title('Accuracy under different noise intensity')
    plt.savefig(os.path.join(images_dir, f'Test_Acc{min_noise}-{max_noise}.png'))
    # plt.show()


def predict_mini(solver, data, mean, intensity):
    num_test = data['X_test'].shape[1]
    batch_mask = np.random.choice(num_test, 25)
    X_test = data['X_test'][:, batch_mask].astype(np.float64)
    Y_test = data['Y_test'][batch_mask]
    noise = np.random.normal(loc=mean, scale=intensity, size=X_test.shape)
    X_test += noise
    avg = np.average(X_test)
    std = np.std(X_test)
    X_test = (X_test - avg) / std
    acc, Y_pred = solver.check_accuracy(X_test, Y_test)

    fig = plt.figure()
    axes = []
    for i in range(25):
        b = X_test[:, i].reshape(28, 28)
        axes.append(fig.add_subplot(5, 5, i + 1))
        subplot_title = f"Label {Y_test[i]}, Pred {Y_pred[i]}"
        # axes[-1].set_title(subplot_title)
        plt.xlabel(subplot_title)
        plt.imshow(b)
    fig.suptitle(f'Intensity{intensity} Acc={acc}')
    fig.tight_layout(pad=0.3)
    plt.savefig(os.path.join(images_dir, f'Mini_Set_Intensity_{intensity}.png'))
    plt.show()


if __name__ == '__main__':
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    img_size = 784
    layer_dims = [img_size, 512, 256, 10]
    activation = ['ReLU', 'ReLU']
    num_classes = 10

    batch_size = 100
    weight_reg = 0
    learning_rate = 0.5
    lr_decay = 1
    verbose = 100

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='load model from')

    train_group = parser.add_argument_group('Train model', 'Training an MLP on MNIST')
    train_group.add_argument('-t', '--train', action='store_true',
                             help='train a model(-f continue training from existing model, train from scratch o.w.)')
    train_group.add_argument('-b', '--batch_size', default=100, type=int, help=f'mini batch size default {batch_size}')
    train_group.add_argument('-w', '--weight_reg', default=0, type=float,
                             help=f'L2 regularization factor default {weight_reg}')
    train_group.add_argument('-e', '--epochs', type=int, help='epochs to train')
    train_group.add_argument('-l', '--learning_rate', default=0.5, type=float,
                             help=f'learning_rate, default {learning_rate}')
    train_group.add_argument('-ld', '--lr_decay', default=1, type=float, help=f'learning rate decay default {lr_decay}')
    train_group.add_argument('-v', '--verbose', default=100, type=int,
                             help=f'print info every verbose time default {verbose}, set 0 to be quiet')
    train_group.add_argument('-d', '--dump', default=f'MNIST-MLP-time_stamp',
                             help='file to dump model')

    predict_group = parser.add_argument_group('Predict dataset', 'Predict on MNIST dataset with pretrained model')
    predict_group.add_argument('-p', '--predict', action='store_true', help='predict on MNIST(-f required), default -s')
    predict_group.add_argument('-m', '--mean', type=int, default=0, help='set noise mean, default is 0')
    group = predict_group.add_mutually_exclusive_group()
    group.add_argument('-n', '--noise', nargs=2, type=int,
                       help='set noise intensity with two integer(min max)(max inclusive) step 1, default is 0 0')
    group.add_argument('-s', '--single', type=int, default=0,
                       help='predict on a mini dataset(25 stochastic samples) with noise intensity(default 0)')

    args = parser.parse_args(sys.argv[1:])

    download_mnist()
    mnist_data = load_mnist()

    if args.train:
        print('Training model...')
        if args.batch_size is None:
            args.batch_size = batch_size
        if args.weight_reg is None:
            args.weight_reg = weight_reg
        if args.learning_rate is None:
            args.learning_rate = learning_rate
        if args.lr_decay is None:
            args.lr_decay = lr_decay
        if args.verbose is None:
            args.verbose = verbose

        if args.epochs is None:
            parser.print_help()
            exit(0)
        mnist_data = standardize(mnist_data)
        if args.file:
            solver = Solver.load_model(args.file, mnist_data, args.batch_size, args.epochs, args.dump,
                                       args.verbose != 0, args.verbose)
        else:
            model = Model(layer_dims, num_classes, activation, args.weight_reg)
            solver = Solver(model, mnist_data, args.learning_rate, args.batch_size, args.epochs, args.dump,
                            args.lr_decay, args.verbose != 0, args.verbose)
        solver.train()
        solver.save_model()
    else:
        print('Predicting data...')
        if args.file is None:
            parser.print_help()
            exit(0)
        solver = Solver.load_model(args.file, mnist_data)
        if args.noise is None:
            predict_mini(solver, mnist_data, args.mean, args.single)
        else:
            predict_with_pretrained(solver, mnist_data, args.mean, args.noise[0], args.noise[1])
