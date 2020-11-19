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
            try:
                current_iteration = len(self.search._results['pipeline_results']) + len(evaluation_results)
                if not self.search._check_stopping_condition(self.search._start, current_iteration):
                    return fitted_pipelines, evaluation_results, []

                if self.search.start_iteration_callback:
                    self.search.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)

                self.log_pipeline(pipeline, current_iteration)
                fitted_pipeline, evaluation_result = self._compute_cv_scores(pipeline, self.search, self.X, self.y)
                fitted_pipelines.append(fitted_pipeline)
                evaluation_results.append(evaluation_result)
            except KeyboardInterrupt:
                pipeline_batch = self.search._handle_keyboard_interrupt(pipeline_batch, pipeline)
                if pipeline_batch == []:
                    return fitted_pipelines, evaluation_results, pipeline_batch
        return fitted_pipelines, evaluation_results, pipeline_batch

    def evaluate_pipeline(self, pipeline, log_pipeline=False):
        try:
            if log_pipeline:
                self.log_pipeline(pipeline)
            if self.search.start_iteration_callback:
                self.search.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
            return self._compute_cv_scores(pipeline, self.search, self.X, self.y)
        except KeyboardInterrupt:
            pipeline_batch = self.search._handle_keyboard_interrupt([], pipeline)
            if pipeline_batch == []:
                return pipeline_batch, []
