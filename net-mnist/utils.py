# -*- coding: utf-8 -*-
import torch
import warnings
import torchvision

from sklearn import metrics
from datetime import datetime
from torchvision import transforms
from sklearn.exceptions import UndefinedMetricWarning

# Import network
from network import *

# Filter scikit-learn metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def plot_dataset(inpt):
    # Falta
    # plt.show()
    pass


def load_dataset(path):
    # Transformation to images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Read data
    trainset = torchvision.datasets.MNIST("data",
                                          train=True,
                                          transform=transform,
                                          download=True)

    testset = torchvision.datasets.MNIST("data",
                                         train=False,
                                         transform=transform,
                                         download=True)

    return trainset, testset


def get_hparams(dictionary):
    hparams = {}
    for key, value in dictionary.items():
        if isinstance(value, int) or \
           isinstance(value, str) or \
           isinstance(value, float) or \
           isinstance(value, list):
            hparams[key] = value
    return hparams


def read_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print('Read weights from {}.'.format(args.checkpoint))

    # Discard hparams
    discard = ['run', 'predict', 'checkpoint', 'summary']

    # Restore past checkpoint
    hparams = checkpoint['hparams']
    for key, value in hparams.items():
        if (key not in discard):
            args.__dict__[key] = value


def restore_checkpoint(args):
    if args.checkpoint == 'none':
        return

    # Load provided checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print('Restored weights from {}.'.format(args.checkpoint))

    # Restore weights
    args.network.load_state_dict(checkpoint['state_dict'])

    if args.predict:
        # To do inference
        args.network.eval()
    else:
        # Read optimizer parameters
        args.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # To continue training
        args.network.train()


def process_checkpoint(loss, global_step, args):
    # check if current batch had best generating fitness
    steps_before_best = 100
    if loss < args.best and global_step > steps_before_best:
        args.best = loss

        # Save best checkpoint
        torch.save({
            'state_dict': args.network.state_dict(),
            'optimizer_state_dict': args.optimizer.state_dict(),
            'hparams': args.hparams,
        }, "checkpoint/best_{}.pt".format(args.run))

        # Write tensorboard statistics
        args.writer.add_scalar('Best/loss', loss, global_step)

    # Save current checkpoint
    torch.save({
        'state_dict': args.network.state_dict(),
        'optimizer_state_dict': args.optimizer.state_dict(),
        'hparams': args.hparams,
    }, "checkpoint/last_{}.pt".format(args.run))


def create_run_name(args):
    run = '{}={}'.format('nw', args.network)
    run += '_{}={}'.format('ds', args.dataset)
    run += '_{}={}'.format('op', args.optimizer)
    run += '_{}={}'.format('ep', args.epochs)
    run += '_{}={}'.format('bs', args.batch_size)
    run += '_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    return run


def calculate_metrics(targets, predictions, args, report=True):
    # Calculate metrics
    avg = 'macro'

    # ignore scikit-learn metrics warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # calculate metrics
        met = {
            'acc': metrics.accuracy_score(targets, predictions),
            'bacc': metrics.balanced_accuracy_score(targets, predictions),
            'prec': metrics.precision_score(targets, predictions, average=avg),
            'rec': metrics.recall_score(targets, predictions, average=avg),
            'f1': metrics.f1_score(targets, predictions, average=avg)}

    # Classification report
    if report:
        # Labels to predict and names
        labels = list(range(0, 10))
        names = [str(elem) for elem in labels]

        # Print classification report
        print(metrics.classification_report(targets, predictions,
                                            labels, names))
    return met
