from pydantic_settings import BaseSettings , SettingsConfigDict



class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = ".env" , env_file_encoding="utf-8")
    MODEL_TEACHER_NAME:str = "google-bert/bert-base-uncased"
    MODEL_STUDENT_NAME:str = "google-bert/bert-base-uncased"
    
    MODEL_PATH_TEACHER:str = "Tienle123/bert-base-uncased-finetuned-emotion"
    MODEL_PATH_STUDENT:str = "Tienle123/distilbert-base-uncased-finetuned-emotion"
    
    PATH_FILE_ONNX:str = "onnx/model.onnx"
    
    DATASET_PATH:str = "dair-ai/emotion"
    
    
    LEARNING_RATE:float = 2e-5
    BATCH_SIZE:int = 64
    ALPHA:float = 0.1
    GAMMA:float = 2.0
    WEIGHT_DECAY:float = 0.01
    EPOCHS:int = 1
    NUMBER_LAYERS:int = 3
    TEMPERATURE:float = 2.0
    BATCH_SIZE_SENTENCE:int = 500
    
    
    
     
    RETURN_ACCURACY:bool = True
    FLAG_TRAINING:bool = True
    FLAG_RETURN_MODEL:bool = False
    
    HUGGINGFACE_API : str | None = None
    
    PATH_MODEL_INPUT:str = "./src/dataset_model/model.onnx"
    PATH_MODEL_OUTPUT:str = "./src/dataset_model/model_quant.onnx"
    
    PATH_FILE_ACCURACY_TEACHER:str = "./src/infrastructure/teacher_accuracy.json"
    PATH_FILE_ACCURACY_STUDENT:str = "./src/infrastructure/student_accuracy.json"

    
    
settings = Settings()
