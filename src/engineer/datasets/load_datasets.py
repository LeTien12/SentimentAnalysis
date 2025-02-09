from datasets import load_dataset , DatasetDict

class LoadDataset:
    def __init__(self, settings:dict) -> None:
        self.dataset_path = settings.DATASET_PATH
    def load_dataset(self) -> DatasetDict:
        return load_dataset(self.dataset_path)

