
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


def _print_percentage(leading_str, num_target, num_total):
    print('{}: {}/{} ({})'.format(
        leading_str,
        num_target,
        num_total,
        num_target / num_total
    ))

# TODO: Move evaluation to analysis
class VarianceEvaluation(MetricEvaluationBase):

    def __init__(self):
        self.result_container = ResultContainer()

    def compute(self, container, exhaustive_search=True):
        """
          Args:
            container: EmbeddingContainer
          Return:
            result_container: ResultContainer
        """
        self.result_container.clear()
        self._inter_intra_variance_measure(container, self.result_container, exhaustive_search)
        return self.result_container

    def _inter_intra_variance_measure(self, container, results, exhaustive_search):
        """
          Args:
            exhaustive_search: boolean
        """
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
            trial_step = 2000
            search_length = num_top2k
            max_search_length = int(0.9 * num_total_instance)
            # TODO: Solve the outside search range problem
            while missed:
                step = (1 + trial) * num_topk if trial < 3 else (1 + trial) * trial_step
                if step > 0.2 * max_search_length:
                    step = max_search_length
                search_length = min(step, max_search_length)
                retrieved_indexes, similarities = agent.search(
                    same_class_embeddings, top_k=search_length, is_similarity=True)
                retrieved_label_ids = container.get_label_by_instance_ids(retrieved_indexes)
                hits = retrieved_label_ids == np.asarray([label_id])

                num_pos_hits = np.sum(hits, axis=1)
                has_missed_pos = num_pos_hits != num_topk
                missed = np.any(has_missed_pos)
                if search_length == max_search_length:
                    print('{} Cannot found within top {}, skip'.format(label_id, max_search_length))
                    missed = False
                if not exhaustive_search:
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

                first_sim = similarities[row, 1]
                last_pos_sim = similarities[row, last_pos_id]
                first_neg_sim = similarities[row, first_neg_id]
                last_neg_sim = similarities[row, last_neg_id]

                top2k_last_pos_sim = similarities[row, top2k_last_pos_id]
                top2k_first_neg_sim = similarities[row, top2k_first_neg_id]
                top2k_last_neg_sim = similarities[row, top2k_last_neg_id]

                margin = last_pos_sim - first_neg_sim
                top2k_margin = top2k_last_pos_sim - top2k_first_neg_sim

                topk_purity = np.sum(top2k_hit_arr[:num_topk + 1]) / (num_topk + 1)
                class_purity = np.sum(hit_arr[:last_pos_id + 1]) / (last_pos_id + 1)

                topk_ap = _compute_ap(top2k_hit_arr[:num_topk + 1])
                class_ap = _compute_ap(hit_arr[:last_pos_id + 1])

                # extend region diversity
                extend_label_ids = ret_label_arr[num_topk:num_top2k]
                num_extend_diversity = len(set(extend_label_ids))

                results.add_event(content={
                    'instance_id': q_id,
                    'label_id': label_id,
                    'ret_ids': ret_id,
                    'ret_label_ids': ret_label_arr,
                    #'ret_label_ids': ret_label_arr[:num_top2k + 1],
                    'first_sim': first_sim,
                    'last_pos_index': last_pos_id,
                    'last_pos_sim': last_pos_sim,
                    'first_neg_index': first_neg_id,
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
                    # 'extend_diversity': num_extend_diversity,
                })

    def analyze(self, container, output_path=None):
        events = self.result_container.events

        print(events)

        """
        label_ids = list(map(int, events.label_id.unique()))
        num_instance_for_label_id = {
            label_id: len(container.get_instance_ids_by_label(label_id)) for label_id in label_ids
        }
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
        """

        print('--' * 20 + ' Overall ' + '--' * 20)
        self._overall_summarization(events, container, output_path)
        print('--' * 20 + ' Type  I ' + '--' * 20)
        self._typeI_summarization(events, container, output_path)
        print('--' * 20 + ' Type II ' + '--' * 20)
        self._typeII_summarization(events, container, output_path)
        print('--' * 49)


    def _overall_summarization(self, events, container, output_path=None):
        inst_ids = container.instance_ids
        lids = container.get_label_by_instance_ids(inst_ids)
        lns = container.get_label_name_by_instance_ids(inst_ids)
        labelmap = {l: n for l, n in zip(lids, lns)}

        missed_events = events[events.class_ap != 1.0]
        _print_percentage('[Overall]: Out of search length events', len(missed_events), len(events))
        _print_percentage('[Overall]: Out of search length classes',
            len(missed_events.label_id.unique()), len(events.label_id.unique()))

        class_ap_thresholds = [0.95, 0.9, 0.85, 0.8, 0.5]
        topk_ap_thresholds = [0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5]
        print(' Events')
        for class_ap_thres in class_ap_thresholds:
            missed_events = events[events.class_ap < class_ap_thres]
            _print_percentage(' - class_ap < {} events'.format(class_ap_thres),
                len(missed_events), len(events))
        for topk_ap_thres in topk_ap_thresholds:
            missed_events = events[events.topk_ap < topk_ap_thres]
            _print_percentage(' - topk_ap < {} events'.format(topk_ap_thres),
                len(missed_events), len(events))
        print(' Classes')
        for class_ap_thres in class_ap_thresholds:
            missed_events = events[events.class_ap < class_ap_thres]
            _print_percentage(' - class_ap < {} classes'.format(class_ap_thres),
                len(missed_events.label_id.unique()), len(events.label_id.unique()))
            """Show the identity names
            missed_ids = list(missed_events.label_id.unique())
            name_str = ', '.join(labelmap.get(_id, '') for _id in missed_ids)
            print(name_str)
            """

        if 'first_sim' in events:
            sim_thres = [1.6, 1.5, 1.45, 1.4]
            print(' First Similarty')
            for thres in sim_thres:
                outlier_query_event = events[events.first_sim < thres]
                _print_percentage(' - similarity < {} events'.format(thres),
                len(outlier_query_event), len(events))

        if 'top2k_last_pos_sim' in events:
            sim_thres = [1.5, 1.45, 1.4]
            print(' Top 2k Last Positive')
            for thres in sim_thres:
                outlier_pos_event = events[events.top2k_last_pos_sim < thres]
                _print_percentage(' - similarity < {} events'.format(thres),
                len(outlier_pos_event), len(events))

        if 'last_pos_sim' in events:
            sim_thres = [1.5, 1.45, 1.4, 0.9, 0.5]
            print(' Last Positive')
            for thres in sim_thres:
                outlier_pos_event = events[events.last_pos_sim < thres]
                _print_percentage(' - similarity < {} events'.format(thres),
                len(outlier_pos_event), len(events))

    def _typeI_summarization(self, events, container, output_path=None):

        events['num_topk'] = events.apply(lambda x: len(x.ret_label_ids) // 2, axis=1)
        # type I by margin
        margin_events = events[events.top2k_margin > 0.0]
        class_margin_events = events[events.class_margin > 0.0]
        # type I by purity
        purity_events = events[events.topk_purity == 1.0]
        # type I by margin but without pure
        margin_not_purity_events = margin_events[margin_events.topk_purity < 1.0]

        print('[By Margin]')
        _print_percentage(' events', len(margin_events), len(events))
        _print_percentage(' class margin events', len(class_margin_events), len(events))

        print(' topk mAP: {}'.format(np.mean(margin_events.topk_ap)))
        print(' topk mean purity: {}'.format(np.mean(margin_events.topk_purity)))
        print(' class mAP: {}'.format(np.mean(margin_events.class_ap)))
        print(' class mean purity: {}'.format(np.mean(margin_events.class_purity)))
        print(' mean top2k margin: {}'.format(np.mean(margin_events.top2k_margin)))

        print(' not pure events: {}/{}'.format(
            len(margin_not_purity_events), len(events)))

        print(' classes: {}/{}'.format(
            len(margin_events.label_id.unique()), len(events.label_id.unique())))
        print(' not pure classes: {}/{}'.format(
            len(margin_not_purity_events.label_id.unique()), len(events.label_id.unique())))

        outliers = margin_not_purity_events[
            margin_not_purity_events.last_pos_index > 2 * margin_not_purity_events.num_topk]
        outlier_ids = outliers.label_id.unique()
        print(' Outlier events: {}/{}'.format(len(outliers), len(events)))
        print(' Outlier classes: {}/{}'.format(len(outlier_ids), len(events.label_id.unique())))
        # TODO: Checkout why topk mAP == 1?
        print(' Outlier topk mAP: {}'.format(np.mean(outliers.topk_ap)))
        print(' Outlier topk mean puirty: {}'.format(np.mean(outliers.topk_purity)))
        print(' Outlier class mAP: {}'.format(np.mean(outliers.class_ap)))

        serious_outlier = margin_not_purity_events[
            margin_not_purity_events.last_pos_index > 10 * margin_not_purity_events.num_topk]
        serious_outlier_ids = serious_outlier.label_id.unique()

        inst_ids = container.instance_ids
        lids = container.get_label_by_instance_ids(inst_ids)
        lns = container.get_label_name_by_instance_ids(inst_ids)
        labelmap = {l: n for l, n in zip(lids, lns)}
        print(' #of serious outlier events: {}/{}'.format(len(serious_outlier), len(margin_not_purity_events)))
        print(' #of serious outlier classes: {}/{}'.format(
            len(serious_outlier.label_id.unique()), len(margin_not_purity_events.label_id.unique())))
        # print names
        #serious_outlier_name_str = ', '.join(labelmap.get(_id, '') for _id in serious_outlier_ids)
        #print(serious_outlier_name_str)

        print('[By Purity]')
        _print_percentage(' events', len(purity_events), len(events))
        # TODO: all instance belong to this category classes
        print(' topk mAP: {}'.format(np.mean(purity_events.topk_ap)))
        print(' class mAP: {}'.format(np.mean(purity_events.class_ap)))
        print(' class mean purity: {}'.format(np.mean(purity_events.class_purity)))
        print(' mean top2k margin: {}'.format(np.mean(purity_events.top2k_margin)))
        print(' mean class margin: {}'.format(np.mean(purity_events.class_margin)))

        if 'first_sim' in events:
            sim_thres = [1.6, 1.5, 1.45, 1.4]
            print(' First Similarty')
            for thres in sim_thres:
                outlier_query_event = purity_events[purity_events.first_sim < thres]
                _print_percentage(' - similarity < {} events'.format(thres),
                len(outlier_query_event), len(events))

        if 'top2k_last_pos_sim' in events:
            sim_thres = [1.5, 1.45, 1.4, 1.3, 0.9]
            print(' Top 2k Last Positive')
            for thres in sim_thres:
                outlier_pos_event = purity_events[purity_events.top2k_last_pos_sim < thres]
                _print_percentage(' - similarity < {} events'.format(thres),
                len(outlier_pos_event), len(events))

        if 'last_pos_sim' in events:
            sim_thres = [1.5, 1.45, 1.4, 1.3, 0.9, 0.5]
            print(' Last Positive')
            for thres in sim_thres:
                outlier_pos_event = purity_events[purity_events.last_pos_sim < thres]
                _print_percentage(' - similarity < {} events'.format(thres),
                len(outlier_pos_event), len(events))

            foundation_events = purity_events[purity_events.last_pos_sim >= 1.5]
            _print_percentage(' - foundation events', len(foundation_events), len(events))
            foundation_classes = []
            label_ids = list(map(int, foundation_events.label_id.unique()))
            num_instance_for_label_id = {
                label_id: len(container.get_instance_ids_by_label(label_id)) for label_id in label_ids}
            for label_id in label_ids:
                num_margin_event = len(foundation_events[foundation_events.label_id == label_id])
                if num_margin_event == num_instance_for_label_id[label_id]:
                    foundation_classes.append(label_id)
            _print_percentage(' - foundation classes', len(foundation_classes), len(events.label_id.unique()))
        

    def _typeII_summarization(self, events, container, output_path=None):

        #inst_ids = container.instance_ids
        #lids = container.get_label_by_instance_ids(inst_ids)
        #lns = container.get_label_name_by_instance_ids(inst_ids)
        #labelmap = {l: n for l, n in zip(lids, lns)}

        # Type II by margin
        margin_events = events[events.top2k_margin <= 0.0]
        # Type II by purity
        purity_events = events[events.topk_purity < 1.0]

        typeII_ap_thresholds = [0.99, 0.95, 0.9, 0.85, 0.8, 0.5]

        print('[By Margin]')
        _print_percentage(' events', len(margin_events), len(events))
        # _print_percentage(' classes', len(margin_events.label_id.unique()), len(events.label_id.unique()))
        print(' topk mAP: {}'.format(np.mean(margin_events.topk_ap)))
        print(' topk mean purity: {}'.format(np.mean(margin_events.topk_purity)))
        print(' class mAP: {}'.format(np.mean(margin_events.class_ap)))
        print(' class mean purity: {}'.format(np.mean(margin_events.class_purity)))

        print(' Events')
        for ap_thres in typeII_ap_thresholds:
            margin_low_ap = margin_events[margin_events.topk_ap < ap_thres]
            print(' - topk AP < {} events: {}'.format(ap_thres, len(margin_low_ap)))
        print(' Classes')
        for ap_thres in typeII_ap_thresholds:
            margin_low_ap = margin_events[margin_events.topk_ap < ap_thres]
            print(' - topk AP < {} classes: {}'.format(ap_thres, len(margin_low_ap.label_id.unique())))

        print('[By Purity]')
        _print_percentage(' events', len(purity_events), len(events))
        print(' topk mAP: {}'.format(np.mean(purity_events.topk_ap)))
        print(' class mAP: {}'.format(np.mean(purity_events.class_ap)))
        print(' class mean purity: {}'.format(np.mean(purity_events.class_purity)))
        # _print_percentage(' classes', len(purity_events.label_id.unique()), len(events.label_id.unique()))

        print(' mean topk purity: {}'.format(np.mean(purity_events.topk_purity)))
        purity_thres = [0.99, 0.95, 0.8, .75, .5, .3]
        for thres in purity_thres:
            topk_purity_level_events = purity_events[purity_events.topk_purity < thres]
            _print_percentage(' - topk purity < {} events'.format(thres),
                len(topk_purity_level_events), len(events))

        print(' Events (Large intra-class variance)')
        for ap_thres in typeII_ap_thresholds:
            purity_low_ap = purity_events[purity_events.topk_ap < ap_thres]
            print(' - topk AP < {} events: {}'.format(ap_thres, len(purity_low_ap)))
        print(' Classes')
        for ap_thres in typeII_ap_thresholds:
            purity_low_ap = purity_events[purity_events.topk_ap < ap_thres]
            print(' - topk AP < {} classes: {}'.format(ap_thres, len(purity_low_ap.label_id.unique())))

        # low_topk_purity_ids = purity_events[purity_events.topk_ap < ap_thres].label_id.unique()
        # name_str = ', '.join(labelmap.get(_id, '') for _id in low_topk_purity_ids)
        # print(name_str)
        #low_topk_ap_inst_ids =  list(purity_events[purity_events.topk_ap < ap_thres].instance_id)
        #print(container.get_filename_strings_by_instance_ids(low_topk_ap_inst_ids))
