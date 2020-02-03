
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import numpy as np

from feature_analyzer.evaluations.evaluation_base import MetricEvaluationBase
from feature_analyzer.data_tools.embedding_container import EmbeddingContainer
from feature_analyzer.data_tools.result_container import ResultContainer
from feature_analyzer.index.agent import IndexAgent


def _compute_ap(bool_arr):
    indices = np.where(bool_arr)[0]
    n_pos = len(indices)
    ap = np.mean((np.arange(n_pos) + 1) / (indices + 1))
    return ap


# TODO: Move evaluation to analysis
class VarianceEvaluation(MetricEvaluationBase):

    def __init__(self):
        self.result_container = ResultContainer()

    def compute(self, container):
        """
          Args:
            container: EmbeddingContainer
          Return:
            result_container: ResultContainer
        """
        self.result_container.clear()
        self._inter_intra_variance_measure(container, self.result_container)
        self.analyze(container)
        return self.result_container

    def _inter_intra_variance_measure(self, container, results):
        instance_ids = container.instance_ids
        all_embeddings = container.get_embedding_by_instance_ids(instance_ids)
        agent = IndexAgent('HNSW', instance_ids, all_embeddings, distance_measure='ip')

        label_ids = list(set(container.label_ids))

        num_total_instance = len(all_embeddings)

        for label_id in tqdm(label_ids):

            # Given class info
            same_class_inst_ids = container.get_instance_ids_by_label_ids(label_id)
            same_class_embeddings = container.get_embedding_by_instance_ids(same_class_inst_ids)

            # instance
            num_topk = len(same_class_inst_ids)
            num_top2k = num_topk * 2

            missed = True
            trial = 1
            trial_step = 100
            search_length = num_top2k
            max_search_length = 2000
            while missed:
                search_length = min((1 + trial) * num_topk, max_search_length)
                retrieved_indexes, similarities = agent.search(
                    same_class_embeddings, top_k=search_length, is_similarity=True)
                retrieved_label_ids = container.get_label_by_instance_ids(retrieved_indexes)
                hits = retrieved_label_ids == np.asarray([label_id])

                num_pos_hits = np.sum(hits, axis=1)
                has_missed_pos = num_pos_hits != num_topk
                missed = np.any(has_missed_pos)
                if search_length == max_search_length:
                    print('Cannot found within top {}, skip'.format(max_search_length))
                    missed = False
                #if missed:
                #    print('{} positives are not retrieved, try again {}'.format(np.sum(has_missed_pos), trial))
                trial += 1

            #print('Done, trail: {}'.format(trial))
            # top k instance
            # TODO: Add to overall
            # topk_hits = hits[:, :num_topk]
            # np.isin(retrieved_indexes[:, :num_inst_same_class], same_class_inst_ids)
            # topk_hit_counts = np.sum(topk_hits, axis=1)
            # topk_miss_counts = np.sum(~topk_hits, axis=1)
            # topk_purities = topk_hit_counts / num_topk
            # topk_same_class_purity = np.mean(topk_purities)

            """
            # center
            class_center_embedding = np.mean(same_class_embeddings, axis=0)
            class_center_fluct = np.mean(np.std(same_class_embeddings, axis=0))
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
                top2k_hit_arr = hit_arr[:num_top2k]
                first_neg_id = np.argmax(~hit_arr)
                last_pos_id = np.where(hit_arr)[-1]
                if last_pos_id.size == 0:
                    # TODO: special case, handle it
                    continue
                last_neg_id = np.where(~hit_arr)[-1][-1]
                last_pos_id = last_pos_id[-1]
                # Three scales
                # - topk
                # - top2k
                # - last positive (for all features)
                top2k_first_neg_id = np.argmax(~top2k_hit_arr)
                top2k_last_pos_id = np.where(top2k_hit_arr)[-1][-1]
                top2k_last_neg_id = np.where(~top2k_hit_arr)[-1][-1]

                last_pos_sim = similarities[row, last_pos_id]
                first_neg_sim = similarities[row, first_neg_id]
                last_neg_sim = similarities[row, last_neg_id]

                top2k_last_pos_sim = similarities[row, top2k_last_pos_id]
                top2k_first_neg_sim = similarities[row, top2k_first_neg_id]
                top2k_last_neg_sim = similarities[row, top2k_last_neg_id]

                margin = last_pos_sim - first_neg_sim
                top2k_margin = top2k_last_pos_sim - top2k_first_neg_sim

                topk_purity = np.sum(top2k_hit_arr[:num_topk]) / num_topk
                class_purity = np.sum(hit_arr[:last_pos_id + 1]) / (last_pos_id + 1)

                topk_ap = _compute_ap(top2k_hit_arr[:num_topk])
                class_ap = _compute_ap(hit_arr[:last_pos_id + 1])

                # extend region diversity
                extend_label_ids = ret_label_arr[num_topk:num_top2k]
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
                    'top2k_last_pos_index': top2k_last_pos_id,
                    'top2k_first_neg_index': top2k_first_neg_id,
                    'top2k_last_pos_sim': top2k_last_pos_sim,
                    'top2k_first_neg_sim': top2k_first_neg_sim,
                    'top2k_last_neg_sim': top2k_last_neg_sim,
                    'top2k_margin': top2k_margin,
                    'topk_ap': topk_ap,
                    'topk_purity': topk_purity,
                    'class_margin': margin,
                    'class_ap': class_ap,
                    'class_purity': class_purity,
                    'extend_diversity': num_extend_diversity,
                })

    def analyze(self, container, output_path=None):
        events = self.result_container.events

        label_ids = list(map(int, events.label_id.unique()))
        num_instance_for_label_id = {
            label_id: len(container.get_instance_ids_by_label(label_id)) for label_id in label_ids
        }

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
        near_recog_margin_event = margin_events[margin_events.top2k_last_pos_sim < ref_threshold]

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

        print('[Type II]: No margin with topK AP=1: {}'.format(len(no_margin_events[no_margin_events.topk_ap == 1])))
        print('[Type II]: No purity with topK AP=1: {}'.format(len(no_purity_events[no_purity_events.topk_ap == 1])))
