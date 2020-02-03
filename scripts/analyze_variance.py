
import os
import sys
import json
from os.path import join, basename, dirname

import time
import numpy as np
import pandas as pd

from feature_analyzer.index.agent import IndexAgent
from feature_analyzer.data_tools.embedding_container import EmbeddingContainer
from feature_analyzer.data_tools.result_container import ResultContainer
from feature_analyzer.evaluations.variance_evaluation import VarianceEvaluation


def main(args):
    embedding_container = EmbeddingContainer()
    result_container = ResultContainer()

    result_container.load(args.result_container_path)
    embedding_container.load(args.embedding_container_path)

    start = time.time()

    variance_analyzer = VarianceEvaluation()
    variance_analyzer.result_container = result_container
    variance_analyzer.analyze(embedding_container)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rc', '--result_container_path', type=str, default=None)
    parser.add_argument('-ec', '--embedding_container_path', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='variance_eval_results')
    args = parser.parse_args()
    main(args)
