from train_model import Learner
from utils.set_seed import setup_seed
from utils.arg_parser import config_param_parser
import torch
import warnings
import os

warnings.filterwarnings("ignore")

def main():
    parser = config_param_parser()
    args = parser.parse_args()
    setup_seed(args.randseed)
    if args.label_name == 'WLR':
        _learner = Learner(args)
        _learner.train()


if __name__=="__main__":
    print('Start ...')
    main()
    print('End ...')