from evalml.automl.engines import EngineBase


class SequentialEngine(EngineBase):
    def __init__(self):
        super().__init__()
        self.name = "Sequential Engine"

    def evaluate_batch(self, pipeline_batch, search_iteration_plot=None):
        super().evaluate_batch()
        fitted_pipelines = []
        evaluation_results = []
        while len(pipeline_batch) > 0:
            pipeline = pipeline_batch.pop()
            try:
                current_iteration = len(self.automl._results['pipeline_results']) + len(evaluation_results)
                if not self.automl._check_stopping_condition(self.automl._start, current_iteration):
                    return fitted_pipelines, evaluation_results, []

                if self.automl.start_iteration_callback:
                    self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)

                self.log_pipeline(pipeline, current_iteration)
                fitted_pipeline, evaluation_result = self._compute_cv_scores(pipeline, self.automl, self.X, self.y)

                if search_iteration_plot:
                    search_iteration_plot.update()

                fitted_pipelines.append(fitted_pipeline)
                evaluation_results.append(evaluation_result)
            except KeyboardInterrupt:
                pipeline_batch = self.automl._handle_keyboard_interrupt(pipeline_batch, pipeline)
                if pipeline_batch == []:
                    return fitted_pipelines, evaluation_results, pipeline_batch
        return fitted_pipelines, evaluation_results, pipeline_batch

    def evaluate_pipeline(self, pipeline, log_pipeline=False, search_iteration_plot=None):
        super().evaluate_pipeline()
        try:
            fitted_pipeline = None
            evaluation_results = None
            if log_pipeline:
                self.log_pipeline(pipeline)
            if self.automl.start_iteration_callback:
                self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
            fitted_pipeline, evaluation_results = self._compute_cv_scores(pipeline, self.automl, self.X, self.y)

            if search_iteration_plot:
                search_iteration_plot.update()

            return fitted_pipeline, evaluation_results
        except KeyboardInterrupt:
            pipeline_batch = self.automl._handle_keyboard_interrupt([], pipeline)
            return pipeline_batch, []
