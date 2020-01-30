

import os
import sys
import json
from os.path import join, basename, dirname

import time
import numpy as np
import pandas as pd

from feature_analyzer.index.agent import IndexAgent
from feature_analyzer.data_tools.embedding_container import EmbeddingContainer
from feature_analyzer.evaluations.variance_evaluation import VarianceEvaluation


def main(args):

    if args.data_dir is None:
        return

    container = EmbeddingContainer()
    container.load(args.data_dir)

    print(container)

    """
    var_eval = VarianceEvaluation(config=ConfigParser({
        'index_agent': 'HNSW',
        'container_size': 10000,
        'database': {
            'database_type': None,
            'database_config': None,
        },
        'chosen_evaluations': ['VarianceEvaluation'],
        'evaluation_options': {
            'VarianceEvaluation': {
                'sampling': {},
                'distance_measure': {},
                'attribute': {},
                'option': {},
            }
        }
    }))
    """
    var_eval = VarianceEvaluation()

    start = time.time()

    res = var_eval.compute(container)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    print(res.events)

    res.save(args.output_dir)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='variance_eval_results')
    parser.add_argument('-m', '--meta_data_path', type=str, default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    parser.add_argument('-es', '--embedding_size', type=int, default=1024)
    args = parser.parse_args()
    main(args)
