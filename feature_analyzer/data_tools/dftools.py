
import pandas as pd
import random


"""
  The given dataframe must have 5 columns:
    'file_name', 'img_path', 'width', 'height', 'bboxes'
"""

def save(df, path):
  # in HDF5 format
  if path.endswith('.h5'):
    with pd.HDFStore(path, 'w') as store:
      store['df'] = df
  elif path.endswith('.feather'):
    df.to_feather(path)
  else:
    raise ValueError('Only .h5 and .feather are supported.')
  return True


def load(path):
  # in HDF5 format
  if path.endswith('.h5'):
    with pd.HDFStore(path, 'r') as store:
      df = store['df']
  elif path.endswith('.feather'):
    df = pd.read_feather(path)
  else:
    raise ValueError('Only .h5 and .feather are supported.')
  return df


def split2(df, ratio=0.8):
  """split dataframe into 2 indep dfs
    Args:
      df: Original
      ratio: a float
    Returns
      df1, df2
  """
  df1 = df.sample(frac=ratio, random_state=1234)
  df2 = df.drop(df1.index)
  return df1, df2


def merge(df1, df2):
  # merge two dataframes to dump the same coco annotation
  df = pd.concat([df1, df2])
  print('Merged df has {} rows from df1 ({}) + df2 ({})'.format(len(df), len(df1), len(df2)))
  return df
