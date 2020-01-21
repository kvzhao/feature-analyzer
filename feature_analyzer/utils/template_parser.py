
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from os.path import join
from os.path import basename

import json
import numpy as np
import pandas as pd

from shutil import copyfile


def load_json(path):
    with open(path, 'r') as fp:
        d = json.load(fp)
    return d

def _create_event(pair_list):
    # Zero-effort implementation
    id_pair = []
    event_pair = []
    id_A, id_B = [], []
    img_A, img_B = [], []
    inst_A, inst_B = [], []
    gt_lm_A, gt_lm_B = [], []
    pd_lm_A, pd_lm_B = [], []
    scores = []
    for pair in pair_list:
        id_a = pair['file'][0].split('/')[0]
        id_b = pair['file'][1].split('/')[0]
        id_pair.append(frozenset([id_a, id_b]))
        event_pair.append(frozenset(pair['file']))
        id_A.append(id_a)
        img_A.append(pair['file'][0].split('/')[1])
        gt_lm_A.append(pair['lm_gt'][0])
        pd_lm_A.append(pair['lm_pred'][0])
        id_B.append(id_b)
        img_B.append(pair['file'][1].split('/')[1])
        inst_A.append(pair['file'][0])
        inst_B.append(pair['file'][1])
        gt_lm_B.append(pair['lm_gt'][1])
        pd_lm_B.append(pair['lm_pred'][1])
        scores.append(pair['score'])
    return pd.DataFrame({
        'id_pair': id_pair,
        'event_pair': event_pair,
        'id_A': id_A,
        'id_B': id_B,
        'img_A': img_A,
        'img_B': img_B,
        'inst_A': inst_A,
        'inst_B': inst_B,
        'gt_lm_A': gt_lm_A,
        'gt_lm_B': gt_lm_B,
        'pd_lm_A': pd_lm_A,
        'pd_lm_B': pd_lm_B,
        'score': scores,
    })


def _create_lm_failure(wronglm_dict):
    event_names = list(wronglm_dict.keys())
    ids = list(map(lambda x: x.split('/')[0], event_names))
    imgs = list(map(lambda x: x.split('/')[1], event_names))
    gt_lms = list(map(lambda x: wronglm_dict[x]['gt'], event_names))
    pd_lms = list(map(lambda x: wronglm_dict[x]['predict'], event_names))
    nmes = list(map(lambda x: wronglm_dict[x]['nme'], event_names))
    
    return pd.DataFrame({
        'event': event_names,
        'id': ids,
        'img': imgs,
        'gt_lm': gt_lms,
        'pd_lm': pd_lms,
        'NME': nmes,
    })

    
class InferenceResult(object):
    def __init__(self, data_dir):
        """Binary template parser
            Several output
            - false_negative_list.json
            - false_positive_list.json
            - no_face_img_list.json
            - wrong_landmark_img_list.json
        """
        self.data_dir = data_dir
        self.fp_list = load_json(join(self.data_dir, 'false_positive_list.json'))
        self.fn_list = load_json(join(self.data_dir, 'false_negative_list.json'))
        self.noface_list = load_json(join(self.data_dir, 'no_face_img_list.json'))
        self.wronglm_dict = load_json(join(self.data_dir, 'wrong_landmark_img_list.json'))
        print('Load from {}'.format(self.data_dir))
        print('FP: {}, FN: {}, NF: {}, WLM:{}'.format(len(self.fp_list), len(self.fn_list), len(self.noface_list), len(self.wronglm_dict)))
        # Convert to dataframe
        self.fp_events = _create_event(self.fp_list)
        self.fn_events = _create_event(self.fn_list)
        self.wronglm = _create_lm_failure(self.wronglm_dict)

# Converter Codes
def _create_anno_table(file_list, pos_pair, neg_pair=None):
    _id_file_map = {
        _id: name for _id, name in enumerate(file_list)
    }
    _id_pos_list = {}
    _id_neg_list = {}

    for pair in tqdm(pos_pair):
        a = pair[0]
        b = pair[1]
        
        if a not in _id_pos_list:
            _id_pos_list[a] = []
        if b not in _id_pos_list:
            _id_pos_list[b] = []
        _id_pos_list[a].append(b)
        _id_pos_list[b].append(a)
    print('Pos table is done.')
 
    for pair in tqdm(neg_pair):
        a = pair[0]
        b = pair[1]
        
        if a not in _id_neg_list:
            _id_neg_list[a] = []
        if b not in _id_neg_list:
            _id_neg_list[b] = []
        _id_neg_list[a].append(b)
        _id_neg_list[b].append(a)
    print('Neg table is done.')

    return _id_file_map, _id_pos_list, _id_neg_list
