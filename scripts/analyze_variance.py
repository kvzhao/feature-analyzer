
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
    num_identities = len(label_ids)
    num_instance_for_label_id = {
        label_id: len(container.get_instance_ids_by_label(label_id)) for label_id in label_ids
    }

    margin_events = events[events.margin >= 0]
    no_margin_events = events[events.margin < 0]

    num_typeI_events = len(margin_events)
    num_typeII_events = len(no_margin_events)

    print('[instance] TypeI : {}, TypeII: {}'.format(num_typeI_events / num_events, num_typeII_events / num_events))

    tpyeI_identities = []
    tpyeII_identities = []
    for label_id in label_ids:
        num_margin_event = len(margin_events[margin_events.label_id == label_id])
        if num_margin_event == num_instance_for_label_id[label_id]:
            tpyeI_identities.append(label_id)
        else:
            # TODO: different levels
            tpyeII_identities.append(label_id)
    num_tpyeI_identities = len(tpyeI_identities)
    num_tpyeII_identities = len(tpyeII_identities)
    print('[indentity] TypeI : {}, TypeII: {}'.format(num_tpyeI_identities/ num_identities, num_tpyeII_identities/ num_identities))



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
