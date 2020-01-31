
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from feature_analyzer.evaluations.evaluation_base import MetricEvaluationBase
from feature_analyzer.data_tools.embedding_container import EmbeddingContainer
from feature_analyzer.data_tools.result_container import ResultContainer
from feature_analyzer.index.agent import IndexAgent


# TODO: Move evaluation to analysis
class VarianceEvaluation(MetricEvaluationBase):

    def __init__(self):
        #super(VarianceEvaluation, self).__init__(config, mode)
        pass

    def compute(self, container):
        """
          Args:
            container: EmbeddingContainer
          Return:
            result_container: ResultContainer
        """
        self.result_container = ResultContainer()

        self._inter_intra_variance_measure(container, self.result_container)

        return self.result_container


    def _inter_intra_variance_measure(self, container, results):
        instance_ids = container.instance_ids
        all_embeddings = container.get_embedding_by_instance_ids(instance_ids)
        agent = IndexAgent('HNSW', instance_ids, all_embeddings, distance_measure='ip')

        label_ids = list(set(container.label_ids))

        for label_id in label_ids:

            # Given class info
            same_class_inst_ids = container.get_instance_ids_by_label_ids(label_id)
            same_class_embeddings = container.get_embedding_by_instance_ids(same_class_inst_ids)
            class_center_embedding = np.mean(same_class_embeddings, axis=0)
            class_center_fluct = np.mean(np.std(same_class_embeddings, axis=0))
            
            # instance
            num_topk = len(same_class_inst_ids)
            num_top2k = num_topk * 2
            retrieved_indexes, similarities = agent.search(
                same_class_embeddings, top_k=num_top2k, is_similarity=True)
            retrieved_label_ids = container.get_label_by_instance_ids(retrieved_indexes)
            hits = retrieved_label_ids == np.asarray([label_id])

            # top k instance
            topk_hits = hits[:, :num_topk]
            #np.isin(retrieved_indexes[:, :num_inst_same_class], same_class_inst_ids)
            topk_hit_counts = np.sum(topk_hits, axis=1)
            topk_miss_counts = np.sum(~topk_hits, axis=1)
            topk_purities = topk_hit_counts / num_topk
            topk_same_class_purity = np.mean(topk_purities)

            """
            # center
            center_retrieved_indexes, center_similarities = agent.search(
                class_center_embedding, top_k=num_top2k, is_similarity=True)
            center_retrieved_label_ids = container.get_label_by_instance_ids(center_retrieved_indexes)
            center_hits = center_retrieved_label_ids == np.asarray([label_id])
            
            # top k center
            topk_center_hits = center_hits[:, :num_topk]
            topk_center_hit_counts = np.sum(topk_center_hits, axis=1)
            topk_center_purities = topk_center_hit_counts / num_topk
            topk_center_same_class_purity = np.mean(topk_center_purities)
            """
            
            
            # top 2k instance
            assert hits.shape == retrieved_label_ids.shape
            for row, (q_id, ret_id, hit_arr, ret_label_arr) in enumerate(
                    zip(same_class_inst_ids, retrieved_indexes, hits, retrieved_label_ids)):
                # for each arr, must have negative and positive (self)
                first_neg_id = np.argmax(~hit_arr)
                prev_pos_id = first_neg_id - 1
                last_pos_id = np.where(hit_arr)[-1]

                if last_pos_id.size == 0:
                    # TODO: special case, handle it
                    continue

                last_neg_id = np.where(~hit_arr)[-1][-1]

                last_pos_id = last_pos_id[-1]
                last_pos_sim = similarities[row, last_pos_id]
                first_neg_sim = similarities[row, first_neg_id]
                last_neg_sim = similarities[row, last_neg_id]
                margin = last_pos_sim - first_neg_sim

                topk_purity = np.sum(hit_arr[:num_topk]) / num_topk
                #topk_purity = topk_purities[row]

                indices = np.where(hit_arr)[0]
                n_pos = len(indices)
                topk_ap = np.mean((np.arange(n_pos) + 1) / (indices + 1))

                # extend region diversity
                extend_label_ids = ret_label_arr[num_topk:]
                num_extend_diversity = len(set(extend_label_ids))

                results.add_event(content={
                    'instance_id': q_id,
                    'label_id': label_id,
                    'ret_ids': ret_id,
                    'ret_label_ids': ret_label_arr,
                    'last_pos_index': last_pos_id,
                    'first_neg_index': first_neg_id,
                    'last_pos_sim': last_pos_sim,
                    'first_neg_sim': first_neg_sim,
                    'last_neg_sim': last_neg_sim,
                    'margin': margin,
                    'topk_purity': topk_purity,
                    'topk_ap': topk_ap,
                    'extend_diversity': num_extend_diversity,
                })