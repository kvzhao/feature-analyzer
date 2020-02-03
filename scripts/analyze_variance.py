
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

    events = results.events

    label_ids = list(map(int, events.label_id.unique()))
    num_instance_for_label_id = {
        label_id: len(container.get_instance_ids_by_label(label_id)) for label_id in label_ids
    }

    # Type I
    # margin case;
    # purity case;
    margin_events = events[events.top2k_margin > 0.0]
    purity_events = events[events.topk_purity == 1.0]
    margin_not_pure_events = margin_events[margin_events.topk_purity < 1.0]
    # Type II
    no_margin_events = events[events.top2k_margin <= 0.0]
    no_purity_events = events[events.topk_purity != 0.0]

    # NOT SURE
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

    print('[Type I] Margin events: {}'.format(len(margin_events) / len(events)))
    print('[Type I] Purity events: {}'.format(len(purity_events) / len(events)))
    print('[Type I] Margin with Nonpure TopK events: {}'.format(len(margin_not_pure_events) / len(events)))
    print('[Type I] Margin class: {}, {}'.format(len(margin_label_ids), len(margin_label_ids) / len(label_ids)))

    print('[Type I] Pure events: {}'.format(len(pure_margin_event) / len(events)))
    print('[Type I] Near boundary events: {}'.format(len(near_recog_margin_event) / len(margin_events)))

    print('[Type II] No margin events: {}'.format(len(no_margin_events) / len(events)))
    print('[Type II] No purity events: {}'.format(len(no_purity_events) / len(events)))

    ap_thres = 0.95
    no_margin_low_ap = no_margin_events[no_margin_events.topk_ap < ap_thres]
    no_purity_low_ap = no_purity_events[no_purity_events.topk_ap < ap_thres]
    print('[Type II] No Margin Low topk AP (<{}): {}'.format(ap_thres, len(no_margin_low_ap) / len(events)))
    print('[Type II] No Purity Low topk AP (<{}): {}'.format(ap_thres, len(no_purity_low_ap) / len(events)))

    ap_thres = 0.99
    no_margin_low_ap = no_margin_events[no_margin_events.topk_ap < ap_thres]
    no_purity_low_ap = no_purity_events[no_purity_events.topk_ap < ap_thres]
    print('[Type II] No Margin Low topk AP (<{}): {}'.format(ap_thres, len(no_margin_low_ap) / len(events)))
    print('[Type II] No Purity Low topk AP (<{}): {}'.format(ap_thres, len(no_purity_low_ap) / len(events)))

    print('[Type II]: No margin with topK AP = 1: {}'.format(len(no_margin_events[no_margin_events.topk_ap == 1])))
    print('[Type II]: No purity with topK AP = 1: {}'.format(len(no_purity_events[no_purity_events.topk_ap == 1])))


def main(args):
    embedding_container = EmbeddingContainer()
    result_container = ResultContainer()

    result_container.load(args.result_container_path)
    embedding_container.load(args.embedding_container_path)

    start = time.time()

    variance_analyzer(embedding_container, result_container)

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
