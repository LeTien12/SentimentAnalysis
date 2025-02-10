from .model_evalute import PerformanceBenchmark
from pathlib import Path
from sklearn.metrics import accuracy_score


class OnnxPerformanceBenchmark(PerformanceBenchmark):
    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
    def compute_size(self):
        size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        return {"size_mb": size_mb}
    def compute_accuracy(self):
        """This overrides the PerformanceBenchmark.compute_accuracy() method"""
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["label"]
            preds.append(self.pipeline.config.label2id[pred])
            labels.append(label)
        accuracy = accuracy_score(preds, labels)
        return accuracy