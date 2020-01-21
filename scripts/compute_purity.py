

import os
import sys
import json
from os.path import join, basename, dirname

from feature_analyzer.index.agent import IndexAgent
from feature_analyzer.data_tools.embedding_container import EmbeddingContainer


def main(args):

    if args.data_dir is None:
        return

    container = EmbeddingContainer()
    container.load(args.data_dir)

    all_embeddings = container.embeddings
    instance_ids = container.instance_ids
    all_embeddings = container.get_embedding_by_instance_ids(instance_ids)
    agent = IndexAgent('HNSW', instance_ids, all_embeddings, distance_measure='ip')

    print(container)

    for label_id in container.label_ids:
        same_class_inst_ids = container.get_instance_ids_by_label_ids(label_id)
        same_class_embeddings = container.get_embedding_by_instance_ids(same_class_inst_ids)
        num_inst_same_class = len(same_class_inst_ids)
        retrieved_indexes, similarities = agent.search(
            same_class_embeddings, top_k=2*num_inst_same_class, is_similarity=True)
        break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='feature-examples/featobj_deepglint_D40kv2_RMG_v1_v1')
    parser.add_argument('-m', '--meta_data_path', type=str, default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    parser.add_argument('-es', '--embedding_size', type=int, default=1024)
    args = parser.parse_args()
    main(args)
