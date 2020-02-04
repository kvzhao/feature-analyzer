import os
import sys
import json
import struct
from os.path import join, basename, dirname

import numpy as np
import pandas as pd
from tabulate import tabulate
from collections import Counter
from feature_analyzer.data_tools.result_container import ResultContainer


def basic_statistics(events):
    """
      Measures:
        - #of Events
        - #of ID pairs
        - High freq. ID pair
        - High freq. instance
    """
    event_pairs = events.event_pair.unique()
    id_pairs = events.id_pair.unique()
    total_ids = set(events.id_A) | set(events.id_B)

    id_pair_counts = events.id_pair.value_counts().to_dict()
    #median_id_counts = np.median(list(id_pair_counts.values()))
    id_pair_count_thres = 5
    high_freq_id_pairs = [k for k, v in id_pair_counts.items() if v > id_pair_count_thres]
    high_freq_events = events[events.id_pair.isin(high_freq_id_pairs)]

    table = tabulate(
        [
            [
                len(event_pairs),
                len(id_pairs),
                len(high_freq_events.id_pair.unique()),
                len(event_pairs) / len(id_pairs),
                len(total_ids)
            ],
        ],
        headers=['#of Events',
                 '#of ID Pairs',
                 '#of High Freq. (>{}) ID Pairs'.format(id_pair_count_thres),
                 'Events / ID Pairs',
                 '#of IDs',
                 ], tablefmt='orgtbl')
    return table


def summarize_fp_events(fp_events):
    table = basic_statistics(fp_events)
    print('[FP]')
    print(table)

    hard_fP_thres = 1.8
    hard_fp_events = fp_events[fp_events.score > hard_fP_thres]
    hard_fp_ids = hard_fp_events.id_pair.unique()
    print(' - #of Hard (>{}) fp events: {}'.format(hard_fP_thres, len(hard_fp_events)))
    print(' - #of Hard (>{}) fp IDs: {}'.format(hard_fP_thres, len(hard_fp_ids)))
    # print(' - %Hard fp events: {}'.format(len(hard_fp_events) / len(fp_events)))


def summarize_fn_events(fn_events):
    table = basic_statistics(fn_events)
    print('[FN]')
    print(table)

    hard_fn_thres = 0.9
    hard_fn_events = fn_events[fn_events.score < hard_fn_thres]
    hard_fn_ids = hard_fn_events.id_pair.unique()
    print(' - #of Hard (<{}) fn events: {}'.format(hard_fn_thres, len(hard_fn_events)))
    print(' - #of Hard (<{}) fn IDs: {}'.format(hard_fn_thres, len(hard_fn_ids)))
    # print(' - %Hard fn events: {}'.format(len(hard_fn_events) / len(fn_events)))


def main(args):
    fp_rc_path = join(args.result_container_folder, 'fp')
    fn_rc_path = join(args.result_container_folder, 'fn')
    lm_rc_path = join(args.result_container_folder, 'lm')
    fp_rc = ResultContainer()
    fn_rc = ResultContainer()
    lm_rc = ResultContainer()
    fp_rc.load(fp_rc_path)
    fn_rc.load(fn_rc_path)
    lm_rc.load(lm_rc_path)

    fp_events = fp_rc.events
    fn_events = fn_rc.events
    lm_events = lm_rc.events

    summarize_fp_events(fp_events)
    summarize_fn_events(fn_events)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rc', '--result_container_folder', type=str,
        default=None)
    parser.add_argument('-m', '--meta_data_path', type=str,
        default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    args = parser.parse_args()
    main(args)
