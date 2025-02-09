from .model_evalute import PerformanceBenchmark
from pathlib import Path

class OnnxPerformanceBenchmark(PerformanceBenchmark):
    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
    def compute_size(self):
        size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        return {"size_mb": size_mb}