#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch

import datasets as ds

from torchvision import datasets, transforms

from models import MAX_SEQUENCE_LENGTH
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import ade_iid, ade_noniid


def num_batches_per_epoch(num_datapoints: int, batch_size: int) -> int:
    """Determine the number of data batches depending on the dataset size

    Args:
        num_datapoints (int): Number of examples present in the dataset
        batch_size (int): Batch size requested by the algorithm

    Returns:
        int: Number of batches to use per epoch
    """
    num_batches, remaining_datapoints = divmod(num_datapoints, batch_size)

    if remaining_datapoints > 0:
        num_batches += 1

    return num_batches


def get_dataset(args, tokenizer=None, max_seq_len=MAX_SEQUENCE_LENGTH):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.task == 'nlp':
        [complete_dataset] = ds.load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification", split=["train"])
        # Rename column.
        complete_dataset = complete_dataset.rename_column("label", "labels")
        complete_dataset = complete_dataset.shuffle(seed=args.seed)
        # Split into train and test sets.
        split_examples = complete_dataset.train_test_split(test_size=args.test_frac)
        train_examples = split_examples["train"]
        test_examples = split_examples["test"]

        # Tokenize training set.
        train_dataset = train_examples.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
            ),
            batched=True,
            remove_columns=["text"],
        )
        train_dataset.set_format(type="torch")

        # Tokenize test set.
        test_dataset = test_examples.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
            ),
            batched=True,
            remove_columns=["text"],
        )
        test_dataset.set_format(type="torch")

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Ade_corpus
            user_groups = ade_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Ade_corpus
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = ade_noniid(train_dataset, args.num_users)

    elif args.task == 'cv':
        if args.dataset == 'cifar':
            data_dir = '../data/cifar/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=apply_transform)

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = cifar_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = cifar_noniid(train_dataset, args.num_users)

        elif args.dataset == 'mnist' or 'fmnist':
            if args.dataset == 'mnist':
                data_dir = '../data/mnist/'
            else:
                data_dir = '../data/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose equal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users)
        else:
            raise NotImplementedError(
                f"""Unrecognized dataset {args.dataset}.
                Options are: `cifar`, `mnist`, `fmnist`.
                """
            )
    else:
        raise NotImplementedError(
            f"""Unrecognised task {args.task}.
            Options are: `nlp` and `cv`.
            """
        )

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    """Print only function that displays experiment details.
    """
    print('\nExperimental details:')
    print(f'    Task      : {args.task}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
