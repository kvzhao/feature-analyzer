"""Registered

  This table recording legal evaluation class.

"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa

# ===========  Evaluations =============
from feature_analyzer.evaluations.evaluation_base import MetricEvaluationBase
from feature_analyzer.evaluations.ranking_evaluation import RankingEvaluation
from feature_analyzer.evaluations.facenet_evaluation import FacenetEvaluation
from feature_analyzer.evaluations.classification_evaluation import ClassificationEvaluation
from feature_analyzer.evaluations.geometric_evaluation import GeometricEvaluation
from feature_analyzer.evaluations.variance_evaluation import VarianceEvaluation

# ===========  Index Agents =============
from feature_analyzer.index.hnsw_agent import HNSWAgent
from feature_analyzer.index.np_agent import NumpyAgent

# ===========  Query Interface =============
from feature_analyzer.query.csv_reader import CsvReader

from feature_analyzer.core.standard_fields import ConfigStandardFields as config_fields
from feature_analyzer.core.standard_fields import EvaluationStandardFields as eval_fields
from feature_analyzer.core.standard_fields import QueryDatabaseStandardFields as query_fields

# NOTICE: Make sure each function passed correctness test.
REGISTERED_EVALUATION_OBJECTS = {
    eval_fields.ranking: RankingEvaluation,
    eval_fields.facenet: FacenetEvaluation,
    eval_fields.classification: ClassificationEvaluation,
    eval_fields.geometric: GeometricEvaluation,
    eval_fields.variance: VarianceEvaluation,
}

EVALUATION_DISPLAY_NAMES = {
    eval_fields.ranking: 'rank',
    eval_fields.facenet: 'pair',
    eval_fields.classification: 'cls',
    eval_fields.geometric: 'geo',
    eval_fields.variance: 'var',
}


REGISTERED_INDEX_AGENT = {
    config_fields.numpy_agent: NumpyAgent,
    config_fields.hnsw_agent: HNSWAgent,
}

REGISTERED_DATABASE_TYPE = {
    query_fields.csv: CsvReader,
}
