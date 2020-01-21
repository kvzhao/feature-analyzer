
import os
import sys
import json
import struct
from os.path import join, basename, dirname

import numpy as np
import pandas as pd

from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer
import metric_learning_evaluator.data_tools.dftools as dftools


def get_binary_files(path):
    bin_files = []
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.bin'):
                    bin_files.append(join(dir_path, file_name))
    return bin_files


def unpack_template(bin_file_path, es=1024):
    """Unpack the file with the following format

       - 1024 dim feature, float32
       - 1 conf, float32
       - 10 landmarks: [eyeLeft, eyeRight, noise, mouseLeft, mouseRight], float32
      Args:
        bin_file_path: string of full path
      Return:
        A dict of datum
      Raise:
        TODO ValueError: open failure
    """
    ret = {}
    if not os.path.isfile(bin_file_path):
        return ret

    with open(join(dirname(bin_file_path), 'finish.json')) as fp:
        content = json.load(fp)
        filename_list = content.get('file_list', None)

    if filename_list is None:
        return ret

    with open(bin_file_path, 'rb') as f:
        binContent = f.read()
        maxface_struct_fmt = '{}f'.format(str(es + 11))
        maxface_struct_len = struct.calcsize(maxface_struct_fmt)
        struct_unpack = struct.Struct(maxface_struct_fmt).unpack_from

        folder_name = dirname(bin_file_path).split('/')[-1]

        for i, file_path in enumerate(filename_list):
            data = struct_unpack(binContent[i * maxface_struct_len: (i + 1) * maxface_struct_len])
            feature = data[:es]
            conf = data[es]
            landmark = data[es + 1:]

            filename = basename(file_path)
            inst_name = join(folder_name, filename)

            ret[inst_name] = {
                'feature': np.asarray(feature, np.float32),
                'conf': conf,
                'landmark': np.asarray(landmark, np.float32),
            }

    return ret


def main(args):

    embedding_size = args.embedding_size
    template_files = get_binary_files(args.template_folder)

    instance_id_to_path = {}
    label_id_to_folder_name = {}

    if args.meta_data_path is not None:
        meta_df = dftools.load(args.meta_data_path)
    meta_df['folder_name'] = meta_df.apply(lambda x: str(x.Class_ID), axis=1)

    for _id, temp_file in enumerate(template_files):
        folder_name = dirname(temp_file).split('/')[-1]
        label_id_to_folder_name[_id] = folder_name
    folder_name_to_label_id = {
        v: k for k, v in label_id_to_folder_name.items()}

    container = EmbeddingContainer(embedding_size=embedding_size,
                                   container_size=200000,
                                   probability_size=1,
                                   landmark_size=10)

    inst_id = 0
    for temp_file in template_files:
        folder_name = temp_file.split('/')[-2]
        ret = unpack_template(temp_file, embedding_size)

        if not ret:
            print('{} is empty, skip'.format(temp_file))

        instance_id_to_path[inst_id] = temp_file

        label_id = folder_name_to_label_id[folder_name]

        id_meta = meta_df[meta_df.folder_name == folder_name]

        label_name = folder_name
        source = ''
        if not id_meta.empty:
            label_name = str(id_meta.Name.values[0])
            source = str(id_meta.source.values[0])

        for inst_name, content in ret.items():

            score = np.asarray([content['conf']], np.float32)

            container.add(
                inst_id,
                label_id=label_id,
                embedding=content['feature'],
                landmark=content['landmark'],
                probability=score,
                filename=inst_name,
                label_name=label_name,
                attribute={
                    'source': source,
                }
            )
            inst_id += 1

            if inst_id % 10000 == 0:
                print('{} features are added'.format(inst_id))

    print(container)

    container.save(args.output_dir)

    inst2path_df = pd.DataFrame.from_dict(instance_id_to_path, orient='index')
    dftools.save(inst2path_df, join(args.output_dir, 'inst2path.h5'))
    print('Save instance to path table.')

    label2folder_df = pd.DataFrame.from_dict(label_id_to_folder_name, orient='index')
    dftools.save(label2folder_df, join(args.output_dir, 'label2folder.h5'))
    print('Save label to folder table.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--template_folder', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='feature-examples/featobj_deepglint_D40kv2_RMG_v1_v1')
    parser.add_argument('-m', '--meta_data_path', type=str, default='/home/kv_zhao/nist-e2e/datasets/meta.h5')
    parser.add_argument('-es', '--embedding_size', type=int, default=1024)
    args = parser.parse_args()
    main(args)
