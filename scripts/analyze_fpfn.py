import os
import sys
import json
import struct
from os.path import join, basename, dirname

import numpy as np
import pandas as pd
from feature_analyzer.data_tools.result_container import ResultContainer


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

    print(lm_events)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rc', '--result_container_folder', type=str,
        default=None)
    parser.add_argument('-m', '--meta_data_path', type=str, default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    args = parser.parse_args()
    main(args)
