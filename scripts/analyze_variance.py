
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


def variance_analyzer(container, results):
    """
      Args:
        container: EmbeddingContinaer
        results: ResultsContainer
      Return:
        report
    """

    print(results.events)
    events = results.events
    num_events = len(events)

    label_ids = list(map(int, events.label_id.unique()))
    num_classes = len(label_ids)
    num_instance_for_label_id = {
        label_id: len(container.get_instance_ids_by_label(label_id)) for label_id in label_ids
    }

    # Type I
    margin_events = events[events.margin > 0]
    # Type II
    no_margin_events = events[events.margin <= 0]


    margin_label_ids = []
    not_all_margin_label_ids = []
    for label_id in label_ids:
        num_margin_event = len(margin_events[margin_events.label_id == label_id])
        if num_margin_event == num_instance_for_label_id[label_id]:
            margin_label_ids.append(label_id)
        else:
            # TODO: different levels
            not_all_margin_label_ids.append(label_id)

    pure_margin_event = margin_events[margin_events.label_id.isin(margin_label_ids)]
    not_pure_margin_event = margin_events[~margin_events.label_id.isin(margin_label_ids)]

    ref_threshold = 1.45
    near_recog_margin_event = margin_events[margin_events.last_pos_sim < ref_threshold]


    num_margin_events = len(margin_events)
    num_no_margin_events = len(no_margin_events)
    num_margin_classes = len(margin_label_ids)
    num_no_margin_classes = len(not_all_margin_label_ids)


    print('[instance] TypeI : {}, TypeII: {}'.format(num_margin_events / num_events, num_no_margin_events / num_events))
    print('[class] TypeI : {}, TypeII: {}'.format(num_margin_classes/ num_classes, num_no_margin_classes/ num_classes))

    print('Pure events: {}'.format(len(pure_margin_event) / len(events)))
    print('Near boundary events: {}'.format(len(near_recog_margin_event) / len(margin_events)))



def main(args):
    embedding_container = EmbeddingContainer()
    result_container = ResultContainer()

    result_container.load(args.result_container_path)
    embedding_container.load(args.embedding_container_path)

    variance_analyzer(embedding_container, result_container)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rc', '--result_container_path', type=str, default=None)
    parser.add_argument('-ec', '--embedding_container_path', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='variance_eval_results')
    args = parser.parse_args()
    main(args)
