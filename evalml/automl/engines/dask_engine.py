from evalml.automl.engines import EngineBase
from dask.distributed import Client, as_completed


class DaskEngine(EngineBase):
    def __init__(self, dask_client=None):
        super().__init__()
        self.name = "Dask Engine"
        self.client = dask_client if dask_client else Client()

    def load_data(self, X, y):
        self.X_future = self.client.scatter(X)
        self.y_future = self.client.scatter(y)
        super().load_data(X, y)

    def evaluate_batch(self, pipeline_batch, search_iteration_plot=None):
        fitted_pipelines = []
        evaluation_results = []
        if self.automl.start_iteration_callback:
            for pipeline in pipeline_batch:
                self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self.automl)
                
        pipeline_futures = self.client.map(self._compute_cv_scores, pipeline_batch, automl=self.automl, X=self.X_future, y=self.y_future)

        for future in as_completed(pipeline_futures):
            pipeline, evaluation_result = future.result()
            self.log_pipeline(pipeline)
            fitted_pipelines.append(pipeline)
            evaluation_results.append(evaluation_result)

        return fitted_pipelines, evaluation_results, []

    def evaluate_pipeline(self, pipeline, log_pipeline=False, search_iteration_plot=None):
        try:
            if log_pipeline:
                self.log_pipeline(pipeline)
            if self.automl.start_iteration_callback:
                self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
            return self._compute_cv_scores(pipeline, self.automl, self.X, self.y)
        except KeyboardInterrupt:
            pipeline_batch = self.automl._handle_keyboard_interrupt([], pipeline)
            if pipeline_batch == []:
                return pipeline_batch, []
