from onnxruntime.quantization import quantize_dynamic, QuantType
from loguru import logger
import subprocess
from huggingface_hub import HfApi


class ModelONNX:
    def __init__(self, settings: dict, session_options=None):
        self.settings = settings
        
    @classmethod
    def run(cls, settings):
        instance = cls(settings)
        instance.run_model_optimize()
        instance.push_model()
        return instance
    
    def run_model_optimize(self) -> None:
        command = [
            "optimum-cli", "export", "onnx",
            "--model", "Tienle123/distilbert-base-uncased-finetuned-emotion",
            "./src/dataset_model/"
        ]
        try:
            logger.info(f"Start model optimization")

            subprocess.run(command, capture_output=True, text=True)
            
            quantize_dynamic(self.settings.PATH_MODEL_INPUT, self.settings.PATH_MODEL_OUTPUT, weight_type=QuantType.QInt8)
            
            logger.info(f"Successful model optimization")

            
            
        except Exception as e:
            logger.error(f"{e}")
            
    def push_model(self):
        api = HfApi(token= self.settings.HUGGINGFACE_API)
        repo_id = self.settings.MODEL_PATH_STUDENT

        try:
            api.upload_file(
                path_or_fileobj=self.settings.PATH_MODEL_OUTPUT,
                path_in_repo=self.settings.PATH_FILE_ONNX,
                repo_id=repo_id,
            )
            logger.info(f"Model uploaded successfully")

        except Exception as e:
            logger.error(f"{e}") 
            