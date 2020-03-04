import numpy as np
from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.components import (
    OneHotEncoder,
    RandomForestClassifier,
    RFClassifierSelectFromModel,
    SimpleImputer
)
from evalml.problem_types import ProblemTypes


class RFBinaryClassificationPipeline(BinaryClassificationPipeline):
    """Random Forest Pipeline for binary classification"""
    name = "Random Forest Classifier w/ One Hot Encoder + Simple Imputer + RF Classifier Select From Model"
    model_type = ModelTypes.RANDOM_FOREST
    problem_types = [ProblemTypes.BINARY]

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, n_estimators, max_depth, impute_strategy,
                 percent_features, number_features, n_jobs=-1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        estimator = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           n_jobs=n_jobs,
                                           random_state=random_state)
        feature_selection = RFClassifierSelectFromModel(n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        number_features=number_features,
                                                        percent_features=percent_features,
                                                        threshold=-np.inf,
                                                        n_jobs=n_jobs,
                                                        random_state=random_state)

        super().__init__(component_list=[enc, imputer, feature_selection, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)