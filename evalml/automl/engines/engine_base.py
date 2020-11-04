import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.logger import get_logger, update_pipeline
from sklearn.model_selection import train_test_split

logger = get_logger(__file__)


class EngineBase(ABC):
    def __init__(self):
        self.name = "Base Engine"
        self.X = None
        self.y = None
        self.search = None

    def load_data(self, X, y):
        self.X = X
        self.y = y

    def load_search(self, search):
        self.search = search

    @abstractmethod
    def evaluate_batch(self, pipeline_batch):
        if not self.X or not self.y:
            raise ValueError("Dataset has not been loaded into the engine. Call load_data(X, y)")

    @abstractmethod
    def evaluate_pipeline(self, pipeline):
        if not self.X or not self.y:
            raise ValueError("Dataset has not been loaded into the engine. Call load_data(X, y)")

    def log_pipeline(self, pipeline, current_iteration=None):
        desc = f"{pipeline.name}"
        if len(desc) > self.search._MAX_NAME_LEN:
            desc = desc[:self.search._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(self.search._MAX_NAME_LEN)

        update_pipeline(logger,
                        desc,
                        (current_iteration if current_iteration else len(self.search._results['pipeline_results'])) + 1,
                        self.search.max_iterations,
                        self.search._start,
                        1 if self.search._automl_algorithm.batch_number == 0 else self.search._automl_algorithm.batch_number,
                        self.search.show_batch_output)

    def _compute_cv_scores(self, pipeline, X, y):
        start = time.time()
        cv_data = []
        logger.info("\tStarting cross validation")
        for i, (train, test) in enumerate(self.search.data_split.split(X, y)):
            if pipeline.model_family == ModelFamily.ENSEMBLE and i > 0:
                # Stacked ensembles do CV internally, so we do not run CV here for performance reasons.
                logger.debug(f"Skipping fold {i} because CV for stacked ensembles is not supported.")
                break
            logger.debug(f"\t\tTraining and scoring on fold {i}")
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            if self.search.problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
                diff_train = set(np.setdiff1d(y, y_train))
                diff_test = set(np.setdiff1d(y, y_test))
                diff_string = f"Missing target values in the training set after data split: {diff_train}. " if diff_train else ""
                diff_string += f"Missing target values in the test set after data split: {diff_test}." if diff_test else ""
                if diff_string:
                    raise Exception(diff_string)
            objectives_to_score = [self.search.objective] + self.search.additional_objectives
            cv_pipeline = None
            try:
                X_threshold_tuning = None
                y_threshold_tuning = None
                if self.search.optimize_thresholds and self.search.objective.problem_type == ProblemTypes.BINARY and self.search.objective.can_optimize_threshold:
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=self.search.random_state)
                cv_pipeline = pipeline.clone(pipeline.random_state)
                logger.debug(f"\t\t\tFold {i}: starting training")
                cv_pipeline.fit(X_train, y_train)
                logger.debug(f"\t\t\tFold {i}: finished training")
                if self.search.objective.problem_type == ProblemTypes.BINARY:
                    cv_pipeline.threshold = 0.5
                    if self.search.optimize_thresholds and self.search.objective.can_optimize_threshold:
                        logger.debug(f"\t\t\tFold {i}: Optimizing threshold for {self.search.objective.name}")
                        y_predict_proba = cv_pipeline.predict_proba(X_threshold_tuning)
                        if isinstance(y_predict_proba, pd.DataFrame):
                            y_predict_proba = y_predict_proba.iloc[:, 1]
                        else:
                            y_predict_proba = y_predict_proba[:, 1]
                        cv_pipeline.threshold = self.search.objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
                        logger.debug(f"\t\t\tFold {i}: Optimal threshold found ({cv_pipeline.threshold:.3f})")
                logger.debug(f"\t\t\tFold {i}: Scoring trained pipeline")
                scores = cv_pipeline.score(X_test, y_test, objectives=objectives_to_score)
                logger.debug(f"\t\t\tFold {i}: {self.search.objective.name} score: {scores[self.search.objective.name]:.3f}")
                score = scores[self.search.objective.name]
            except Exception as e:
                if isinstance(e, PipelineScoreError):
                    logger.info(f"\t\t\tFold {i}: Encountered an error scoring the following objectives: {', '.join(e.exceptions)}.")
                    logger.info(f"\t\t\tFold {i}: The scores for these objectives will be replaced with nan.")
                    logger.info(f"\t\t\tFold {i}: Please check {logger.handlers[1].baseFilename} for the current hyperparameters and stack trace.")
                    logger.debug(f"\t\t\tFold {i}: Hyperparameters:\n\t{pipeline.hyperparameters}")
                    logger.debug(f"\t\t\tFold {i}: Exception during automl search: {str(e)}")
                    nan_scores = {objective: np.nan for objective in e.exceptions}
                    scores = {**nan_scores, **e.scored_successfully}
                    scores = OrderedDict({o.name: scores[o.name] for o in [self.search.objective] + self.search.additional_objectives})
                    score = scores[self.search.objective.name]
                else:
                    logger.info(f"\t\t\tFold {i}: Encountered an error.")
                    logger.info(f"\t\t\tFold {i}: All scores will be replaced with nan.")
                    logger.info(f"\t\t\tFold {i}: Please check {logger.handlers[1].baseFilename} for the current hyperparameters and stack trace.")
                    logger.debug(f"\t\t\tFold {i}: Hyperparameters:\n\t{pipeline.hyperparameters}")
                    logger.debug(f"\t\t\tFold {i}: Exception during automl search: {str(e)}")
                    score = np.nan
                    scores = OrderedDict(zip([n.name for n in self.search.additional_objectives], [np.nan] * len(self.search.additional_objectives)))

            ordered_scores = OrderedDict()
            ordered_scores.update({self.search.objective.name: score})
            ordered_scores.update(scores)
            ordered_scores.update({"# Training": len(y_train)})
            ordered_scores.update({"# Testing": len(y_test)})

            evaluation_entry = {"all_objective_scores": ordered_scores, "score": score, 'binary_classification_threshold': None}
            if isinstance(cv_pipeline, BinaryClassificationPipeline) and cv_pipeline.threshold is not None:
                evaluation_entry['binary_classification_threshold'] = cv_pipeline.threshold
            cv_data.append(evaluation_entry)
        training_time = time.time() - start
        cv_scores = pd.Series([fold['score'] for fold in cv_data])
        cv_score_mean = cv_scores.mean()
        logger.info(f"\tFinished cross validation - mean {self.search.objective.name}: {cv_score_mean:.3f}")
        return {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_score_mean}
