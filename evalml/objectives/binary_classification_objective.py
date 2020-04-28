import pandas as pd
from scipy.optimize import minimize_scalar

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class BinaryClassificationObjective(ObjectiveBase):
    """
    Base class for all binary classification objectives.

    problem_type (ProblemTypes): Specifies the type of problem this objective is defined for (binary classification)
    can_optimize_threshold (bool): Determines if threshold used by objective can be optimized or not.
    """
    problem_type = ProblemTypes.BINARY

    @property
    def can_optimize_threshold(cls):
        """Returns a boolean determining if we can optimize the binary classification objective threshold. This will be false for any objective that works directly with predicted probabilities, like log loss and AUC. Otherwise, it will be true."""
        return not cls.score_needs_proba

    def optimize_threshold(self, ypred_proba, y_true, X=None):
        """Learn a binary classification threshold which optimizes the current objective.

        Arguments:
            ypred_proba (list): The classifier's predicted probabilities

            y_true (list): The ground truth for the predictions.

            X (pd.DataFrame, optional): Any extra columns that are needed from training data.

        Returns:
            Optimal threshold for this objective
        """
        if not self.can_optimize_threshold:
            raise RuntimeError("Trying to optimize objective that can't be optimized!")

        def cost(threshold):
            y_predicted = self.decision_function(ypred_proba=ypred_proba, threshold=threshold, X=X)
            cost = self.objective_function(y_true, y_predicted, X=X)
            return -cost if self.greater_is_better else cost

        optimal = minimize_scalar(cost, method='Golden', options={"maxiter": 100})
        return optimal.x

    def decision_function(self, ypred_proba, threshold=0.5, X=None):
        """Apply a learned threshold to predicted probabilities to get predicted classes.

        Arguments:
            ypred_proba (list): The classifier's predicted probabilities

            threshold (float, optional): Threshold used to make a prediction. Defaults to 0.5.

            X (pd.DataFrame, optional): Any extra columns that are needed from training data.

        Returns:
            predictions
        """
        if not isinstance(ypred_proba, pd.Series):
            ypred_proba = pd.Series(ypred_proba)
        return ypred_proba > threshold