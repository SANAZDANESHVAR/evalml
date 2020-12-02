import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)
from evalml.utils.logger import get_logger, update_pipeline

logger = get_logger(__file__)


class EngineBase(ABC):
    def __init__(self):
        self.name = "Base Engine"
        self.X = None
        self.y = None
        self.loaded_search_info = False

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def load_search(self, search_obj):
        self.problem_type = search_obj.problem_type
        self.objective = search_obj.objective
        self.additional_objectives = search_obj.additional_objectives
        self.optimize_thresholds = search_obj.optimize_thresholds
        self.random_state = search_obj.random_state
        self.error_callback = search_obj.error_callback
        self.data_split = search_obj.data_split

        self._MAX_NAME_LEN = search_obj._MAX_NAME_LEN
        self.max_iterations = search_obj.max_iterations
        self._start = search_obj._start
        self.show_batch_output = search_obj.show_batch_output
        self.start_iteration_callback = search_obj.start_iteration_callback
        self._handle_keyboard_interrupt = search_obj._handle_keyboard_interrupt
        self._check_stopping_condition = search_obj._check_stopping_condition
        self.search = search_obj # Less than ideal but needed for the 

        self.update_search_progress(search_obj)
        self.loaded_search_info = True

    def update_search_progress(self, search_obj):
        self._automl_algorithm = search_obj._automl_algorithm
        self._results = search_obj._results

    @abstractmethod
    def evaluate_batch(self, pipeline_batch=None):
        if self.X is None or self.y is None:
            raise ValueError("Dataset has not been loaded into the engine. Call `load_data` with training data.")

        if not self.loaded_search_info:
            raise ValueError("Search info has not been loaded into the engine. Call `load_search` with search context.")

    @abstractmethod
    def evaluate_pipeline(self, pipeline=None):
        if self.X is None or self.y is None:
            raise ValueError("Dataset has not been loaded into the engine. Call `load_data` with training data.")

        if not self.loaded_search_info:
            raise ValueError("Search info has not been loaded into the engine. Call `load_search` with search context.")

    def log_pipeline(self, pipeline, current_iteration=None):
        desc = f"{pipeline.name}"
        if len(desc) > self._MAX_NAME_LEN:
            desc = desc[:self._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(self._MAX_NAME_LEN)

        update_pipeline(logger,
                        desc,
                        (current_iteration if current_iteration else len(self._results['pipeline_results'])) + 1,
                        self.max_iterations,
                        self._start,
                        1 if self._automl_algorithm.batch_number == 0 else self._automl_algorithm.batch_number,
                        self.show_batch_output)

    def _compute_cv_scores(self, pipeline, X, y):
        start = time.time()
        cv_data = []
        logger.info("\tStarting cross validation")
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)

        X_pd = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        for i, (train, test) in enumerate(self.data_split.split(X_pd, y_pd)):

            if pipeline.model_family == ModelFamily.ENSEMBLE and i > 0:
                # Stacked ensembles do CV internally, so we do not run CV here for performance reasons.
                logger.debug(f"Skipping fold {i} because CV for stacked ensembles is not supported.")
                break
            logger.debug(f"\t\tTraining and scoring on fold {i}")
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            if self.problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
                diff_train = set(np.setdiff1d(y.to_series(), y_train.to_series()))
                diff_test = set(np.setdiff1d(y.to_series(), y_test.to_series()))
                diff_string = f"Missing target values in the training set after data split: {diff_train}. " if diff_train else ""
                diff_string += f"Missing target values in the test set after data split: {diff_test}." if diff_test else ""
                if diff_string:
                    raise Exception(diff_string)
            objectives_to_score = [self.objective] + self.additional_objectives
            cv_pipeline = None
            try:
                X_threshold_tuning = None
                y_threshold_tuning = None
                if self.optimize_thresholds and self.objective.is_defined_for_problem_type(ProblemTypes.BINARY) and self.objective.can_optimize_threshold:
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state)
                cv_pipeline = pipeline.clone(pipeline.random_state)
                logger.debug(f"\t\t\tFold {i}: starting training")
                cv_pipeline.fit(X_train, y_train)
                logger.debug(f"\t\t\tFold {i}: finished training")
                if self.objective.is_defined_for_problem_type(ProblemTypes.BINARY):
                    cv_pipeline.threshold = 0.5
                    if self.optimize_thresholds and self.objective.can_optimize_threshold:
                        logger.debug(f"\t\t\tFold {i}: Optimizing threshold for {self.objective.name}")
                        y_predict_proba = cv_pipeline.predict_proba(X_threshold_tuning)
                        if isinstance(y_predict_proba, pd.DataFrame):
                            y_predict_proba = y_predict_proba.iloc[:, 1]
                        else:
                            y_predict_proba = y_predict_proba[:, 1]
                        cv_pipeline.threshold = self.objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
                        logger.debug(f"\t\t\tFold {i}: Optimal threshold found ({cv_pipeline.threshold:.3f})")
                logger.debug(f"\t\t\tFold {i}: Scoring trained pipeline")
                scores = cv_pipeline.score(X_test, y_test, objectives=objectives_to_score)
                logger.debug(f"\t\t\tFold {i}: {self.objective.name} score: {scores[self.objective.name]:.3f}")
                score = scores[self.objective.name]
            except Exception as e:
                if self.error_callback is not None:
                    self.error_callback(exception=e, traceback=traceback.format_tb(sys.exc_info()[2]), automl=self.search,
                                        fold_num=i, pipeline=pipeline)
                if isinstance(e, PipelineScoreError):
                    nan_scores = {objective: np.nan for objective in e.exceptions}
                    scores = {**nan_scores, **e.scored_successfully}
                    scores = OrderedDict({o.name: scores[o.name] for o in [self.objective] + self.additional_objectives})
                    score = scores[self.objective.name]
                else:
                    score = np.nan
                    scores = OrderedDict(zip([n.name for n in self.additional_objectives], [np.nan] * len(self.additional_objectives)))

            ordered_scores = OrderedDict()
            ordered_scores.update({self.objective.name: score})
            ordered_scores.update(scores)
            ordered_scores.update({"# Training": y_train.shape[0]})
            ordered_scores.update({"# Testing": y_test.shape[0]})

            evaluation_entry = {"all_objective_scores": ordered_scores, "score": score, 'binary_classification_threshold': None}
            if isinstance(cv_pipeline, BinaryClassificationPipeline) and cv_pipeline.threshold is not None:
                evaluation_entry['binary_classification_threshold'] = cv_pipeline.threshold
            cv_data.append(evaluation_entry)
        training_time = time.time() - start
        cv_scores = pd.Series([fold['score'] for fold in cv_data])
        cv_score_mean = cv_scores.mean()
        logger.info(f"\tFinished cross validation - mean {self.objective.name}: {cv_score_mean:.3f}")
        return pipeline, {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_score_mean}
