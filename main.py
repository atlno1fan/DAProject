from Bert4Rec_cold_start import *
from Peter4Rec_cold_start import *
from rnn_cold_start import *
from knn_cold_start import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert')
    args = parser.parse_args()
    if args.model == 'bert':
        run_experiment_bert4rec()
    elif args.model == 'peter4rec':
        run_experiment_peter4rec()
    elif args.model == 'rnn':
        run_experiment_rnn()
    elif args.model == 'knn':
        run_experiment_knn()
