import argparse
from src.data.data_loader import VerseDataset
from src.models.train_model import train_u_net
from src.models.predict_model import predict_test_set
import os



def handle_arguments():
    """Handles console arguments. 'python main.py --h' prompts an overview

    :returns
        prediction      = [option] switch on prediction mode
        image-dir       = [str] string containing path to image dir, compulsory
        model           = [str] string containing image path, optional
        epochs          = [int] number of iterations over the training data
        frac split      = [float] train test split fraction
        seed            = [int] seed for random operation...
        batch_size      = [int] batch size for training
        steps_per_epoch = [int] steps per epoch
    """
    # process the command options
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction', action='store_true', help='when switched on, '
                                                                           'do a prediction on an image and a given model, '
                                                                           'default model is most recent one')
    #TODO: subparser usage
    parser.add_argument('-id', '--image', type=str, default='', help='path to image-dir you want predictions on')
    parser.add_argument('-m', '--model', type=str, default='default', help='path to model you want to use')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number '
                                                                     'of iterations over training data, default: 50')
    parser.add_argument('-f', '--fraction', type=float, default=0.8, help='fraction '
                                                                          'of train test split, default 0.8')
    parser.add_argument('-s', '--seed', type=int, default=2020, help='seed '
                                                                     'to give to random function, default 2020')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size for training, default: 8')
    parser.add_argument('-se', '--steps_per_epoch', type=int, default=None, help='steps per epoch for training, default: None')
    parser.add_argument('-d', '--debug', type=int, default=False, help='1 for debug mode, else leave empty, default: False')
    
    input_args = parser.parse_args()
    return input_args


if __name__ == '__main__':
    # get console arguments
    args = handle_arguments()

    if (args.debug == 1):
        debug = True
    else: 
        debug = False
        
    # set paths
    BASE_PATH = 'data/raw/training_data'
    BASE_PATH_NORMALIZED = 'data/processed/normalized-images'

    if args.prediction:
        # use main to predict a serie of test files. See predict_test_set doc-string for more info
        assert args.image != '', "no path prediction image, use -im"
        # get model dir
        image_list = os.listdir(args.image)
        if args.model == 'default':
            predict_model = 'models/default.h5'
        else:
            predict_model = args.model
        # predict
        predict_test_set(predict_model, image_list)
        # find saved files in 'models/predictions'
    else:
        # generate training and validation data
        verse_dataset = VerseDataset(base_path=BASE_PATH_NORMALIZED, seed=args.seed, split=args.fraction, debug=debug)

        training_dataset = verse_dataset.get_dataset('train').batch(args.batch_size)
        validation_dataset = verse_dataset.get_dataset('validation').batch(args.batch_size)

        train_u_net(training_dataset, validation_dataset, args.epochs, args.steps_per_epoch)
    
    print("--DONE--")
