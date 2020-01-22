

import os
import sys
import json
from os.path import join, basename, dirname

import numpy as np
import pandas as pd

from feature_analyzer.index.agent import IndexAgent
from feature_analyzer.data_tools.embedding_container import EmbeddingContainer


def main(args):

    if args.data_dir is None:
        return

    container = EmbeddingContainer()
    container.load(args.data_dir)

    instance_ids = container.instance_ids
    all_embeddings = container.get_embedding_by_instance_ids(instance_ids)
    agent = IndexAgent('HNSW', instance_ids, all_embeddings, distance_measure='ip')

    print(container)

    mean_purity = 0.0
    impurity_counts = 0.0

    for label_id in container.label_ids:
        same_class_inst_ids = container.get_instance_ids_by_label_ids(label_id)
        same_class_embeddings = container.get_embedding_by_instance_ids(same_class_inst_ids)
        num_inst_same_class = len(same_class_inst_ids)
        retrieved_indexes, similarities = agent.search(
            same_class_embeddings, top_k = 2 * num_inst_same_class, is_similarity=True)

        retrieved_label_ids = container.get_label_by_instance_ids(retrieved_indexes)

        # top k purity
        hits = np.isin(retrieved_indexes[:, :num_inst_same_class], same_class_inst_ids)
        hit_count_each_inst = np.sum(hits, axis=1)
        purity_each_inst = hit_count_each_inst / num_inst_same_class
        same_class_purity = np.mean(purity_each_inst)

        # last positive, first negative
        hit_label_ids = retrieved_label_ids == np.asarray([label_id])
        positive_ids = np.where(hit_label_ids)
        negative_ids = np.where(~hit_label_ids)
        first_negative_ids = np.argmin(hit_label_ids, axis=1)

        print(np.sum(purity_each_inst != 1.0))

        # if first_neg > last_pos (purity == 1.0) => compute margin
        # otherwise, count how many different classes within.

        break

        mean_purity += same_class_purity
    print('Puirty: {}'.format(mean_purity / len(container.label_ids)))
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='feature-examples/featobj_deepglint_D40kv2_RMG_v1_v1')
    parser.add_argument('-m', '--meta_data_path', type=str, default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    parser.add_argument('-es', '--embedding_size', type=int, default=1024)
    args = parser.parse_args()
    main(args)
