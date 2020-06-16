import argparse
from src.data.data_loader import VerseDataset
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
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size for training, default: 32')

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
    verse_dataset = VerseDataset(base_path=BASE_PATH_NORMALIZED, seed=args.seed, split=args.fraction)
    
    training_dataset = verse_dataset.get_dataset('train').batch(args.batch_size)
    validation_dataset = verse_dataset.get_dataset('validation').batch(args.batch_size)

    train_u_net(training_dataset, validation_dataset, args.epochs)
    
    print("--DONE--")
