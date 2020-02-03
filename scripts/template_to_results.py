import os
import sys
import json
import time
import struct
from os.path import join, basename, dirname

import numpy as np
import pandas as pd

import feature_analyzer.data_tools.dftools as dftools
from feature_analyzer.data_tools.result_container import ResultContainer


def load_json(path):
    with open(path, 'r') as fp:
        d = json.load(fp)
    return d


def create_event(pair_list, meta=None):
    id_pair = []
    event_pair = []
    id_A, id_B = [], []
    id_A_name, id_B_name = [], []
    img_A, img_B = [], []
    inst_A, inst_B = [], []
    gt_lm_A, gt_lm_B = [], []
    pd_lm_A, pd_lm_B = [], []
    scores = []

    if meta is None:
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
    else:
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

            id_a_name = meta[meta.Class_ID == id_a].Name
            id_b_name = meta[meta.Class_ID == id_b].Name

            id_a_name = str(id_a_name.values[0]) if not id_a_name.empty else ''
            id_b_name = str(id_b_name.values[0]) if not id_b_name.empty else ''

            id_a_name = id_a_name.replace('"', '')
            id_b_name = id_b_name.replace('"', '')

            id_A_name.append(id_a_name)
            id_B_name.append(id_b_name)

    if meta is not None:
        return pd.DataFrame({
            'id_pair': id_pair,
            'event_pair': event_pair,
            'id_A': id_A,
            'id_B': id_B,
            'id_A_name': id_A_name,
            'id_B_name': id_B_name,
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


def create_lm_failure(wronglm_dict):
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


def main(args):

    if args.template_folder is None:
        return
    start = time.time()
    fp_list = load_json(join(args.template_folder, 'false_positive_list.json'))
    fn_list = load_json(join(args.template_folder, 'false_negative_list.json'))
    noface_list = load_json(join(args.template_folder, 'no_face_img_list.json'))
    wronglm_dict = load_json(join(args.template_folder, 'wrong_landmark_img_list.json'))

    meta = dftools.load(args.meta_data_path)

    os.makedirs(args.result_container_folder, exist_ok=True)

    fp_events = create_event(fp_list, meta)
    print('fp events created.')
    fn_events = create_event(fn_list, meta)
    print('fn events created.')
    lm_events = create_lm_failure(wronglm_dict)
    print('lm events created.')

    fp_rc = ResultContainer()
    fn_rc = ResultContainer()
    lm_rc = ResultContainer()

    fp_rc.events = fp_events
    fn_rc.events = fn_events
    lm_rc.events = lm_events

    fp_rc_path = join(args.result_container_folder, 'fp')
    fn_rc_path = join(args.result_container_folder, 'fn')
    lm_rc_path = join(args.result_container_folder, 'lm')
    fp_rc.save(fp_rc_path)
    fn_rc.save(fn_rc_path)
    lm_rc.save(lm_rc_path)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--template_folder', type=str, default=None)
    parser.add_argument('-rc', '--result_container_folder', type=str,
        default=None)
    parser.add_argument('-m', '--meta_data_path', type=str, default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    args = parser.parse_args()
    main(args)
