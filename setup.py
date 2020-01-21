import os
import sys
from setuptools import setup, find_packages

VERSION = '0.1.0'

EVAL_APP_NAME = 'featureAnalyzer'
ROOT_FOLDER = 'metric_learning_evaluator'
APP_FOLDER = 'application'
AUTHORS = ''


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


install_requires = [
    'pyyaml',
    'numpy',
    'scipy',
    'sklearn',
    'pandas',
    'pytablewriter',
]

eval_setup_info = dict(
    name='feature_analyzer',
    author=AUTHORS,
    version=VERSION,
    description='FeatureAnalyzer',
    long_discription=read('README.md'),
    license='BSD',
    url='https://github.com/kvzhao/feature-analyzer/',
    install_requires=install_requires,
    include_package_data=True,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            '{app_name} = {root_folder}.{app_folder}.evaluation_app:main'.format(
                app_name=EVAL_APP_NAME,
                root_folder=ROOT_FOLDER,
                app_folder=APP_FOLDER)
        ],
    },
)
# Install evaluation
setup(**eval_setup_info)
