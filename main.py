import argparse
import os
from src.data.data_loader import generate_train_validation_test_split, get_dataset_from_generator, \
    testing_data_generator, validation_data_generator, training_data_generator
from datetime import datetime

from src.models.train_model import train_u_net


def handle_arguments():
    """Handles console arguments. 'python main.py --h' promts an overview

    :returns
        epochs      = [int] number of iterations over the training data
        frac split  = [float] train test split fraction
        seed        = [int] seed for random operation...
    """
    # process the command options
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number '
                                                                     'of iterations over training data, default: 50')
    parser.add_argument('-f', '--fraction', type=float, default=0.8, help='fraction '
                                                                          'of train test split, default 0.8')
    parser.add_argument('-s', '--seed', type=int, default=2020, help='seed '
                                                                     'to give to random function, default 2020')

    # parse and print arguments
    args = parser.parse_args()
    #for arg in vars(args):
        #print(f'{arg.upper()}: {getattr(args, arg)}')

    return args


if __name__ == '__main__':
    # get console arguments
    args = handle_arguments()

    # set paths
    BASE_PATH = 'data/raw/training_data'
    BASE_PATH_NORMALIZED = 'data/processed/normalized-images'

    # generate training and validation data
    generate_train_validation_test_split(args.seed, args.fraction)

    # use GPU if available
    # TODO: if Keras does not do this automatically, implement it!

    # make datasets
    training_dataset = get_dataset_from_generator(training_data_generator)
    validation_dataset = get_dataset_from_generator(validation_data_generator)
    testing_dataset = get_dataset_from_generator(testing_data_generator)

    # call the model
    training_dataset = training_dataset.batch(1)
    validation_dataset = validation_dataset.batch(1)

    train_u_net(training_dataset, validation_dataset, 1)
    # train_u_net(training_dataset, validation_dataset, args.epoch)

    print("--DONE--")
