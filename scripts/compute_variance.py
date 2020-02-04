

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

    if args.embedding_container_path is None:
        return

    container = EmbeddingContainer()
    container.load(args.embedding_container_path)

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

    res = var_eval.compute(container, 
        exhaustive_search=args.deep_search)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    res.save(args.result_container_path)

    var_eval.analyze(container)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ec', '--embedding_container_path', type=str, default=None)
    parser.add_argument('-rc', '--result_container_path', type=str, default='variance_eval_results')
    parser.add_argument('-m', '--meta_data_path', type=str, default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    parser.add_argument('-es', '--embedding_size', type=int, default=1024)
    parser.add_argument('--deep_search', dest='deep_search', action='store_true')
    args = parser.parse_args()
    main(args)
