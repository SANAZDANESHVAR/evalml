from evalml.automl.engines import EngineBase


class SequentialEngine(EngineBase):
    def __init__(self):
        super().__init__()
        self.name = "Sequential Engine"

    def evaluate_batch(self, pipeline_batch):
        fitted_pipelines = []
        evaluation_results = []
        while len(pipeline_batch) > 0:
            pipeline = pipeline_batch.pop()
            current_iteration = len(self.search._results['pipeline_results']) + len(evaluation_results)
            if not self.search._check_stopping_condition(self.search._start, current_iteration):
                return fitted_pipelines, evaluation_results
            self.log_pipeline(pipeline, current_iteration)
            evaluation_result = self._compute_cv_scores(pipeline, self.X, self.y)
            fitted_pipelines.append(pipeline)
            evaluation_results.append(evaluation_result)
        return fitted_pipelines, evaluation_results

    def evaluate_pipeline(self, pipeline):
        self.log_pipeline(pipeline)
        fitted_pipeline, evaluation_result = self.search._compute_cv_scores(pipeline, self.X, self.y)
        return [fitted_pipeline], [evaluation_result]
