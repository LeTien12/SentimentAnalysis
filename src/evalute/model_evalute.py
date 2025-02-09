import numpy as np
from pathlib import Path
from time import perf_counter
import torch
from sklearn.metrics import accuracy_score
from datasets import DatasetDict


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset:DatasetDict ,optim_type:str):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        """This overrides the PerformanceBenchmark.compute_accuracy() method"""
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["label"]
            preds.append(self.pipeline.model.config.label2id[pred])
            labels.append(label)
        accuracy = accuracy_score(preds, labels)
        return {"accuracy" : accuracy}

    def compute_size(self):
        """This overrides the PerformanceBenchmark.compute_size() method"""
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        tmp_path.unlink()
        return {"size_mb": round(size_mb,2)}

    def time_pipeline(self, query="The door creaked open by itself, and a cold whisper sent chills down my spine."):
        """This overrides the PerformanceBenchmark.time_pipeline() method"""
        latencies = []
        # Warmup
        for _ in range(10):
            _ = self.pipeline(query)
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(query)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}
        
    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.compute_accuracy())
        metrics[self.optim_type].update(self.time_pipeline())
        return metrics