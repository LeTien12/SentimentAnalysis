from onnxruntime import InferenceSession
from scipy.special import softmax
import numpy as np
from transformers import AutoTokenizer  , AutoConfig 
from huggingface_hub import hf_hub_download




class OnnxPipeline:
    def __init__(self, settings:dict , session_options=None):
        self.setting = settings
        self.model_name = settings.MODEL_PATH_STUDENT
        self.tokenizer = self.load_tokenizer()
        self.config = self.load_config()
        self.model_path = hf_hub_download(repo_id=self.model_name, 
                                     filename=settings.PATH_FILE_ONNX)
        
        self.session = InferenceSession(self.model_path, sess_options=session_options)
        self.output_names = [output.name for output in self.session.get_outputs()]

    def __call__(self, queries):
        inputs = self.tokenizer(queries, return_tensors="np")
        logits = self.session.run(output_names=self.output_names, input_feed=dict(inputs))[0]
        probs = softmax(logits)
        predicted_class_index = np.argmax(probs)
        predicted_label = self.config.id2label[predicted_class_index]
        return [{"label": predicted_label, "score": np.max(probs)}]
    
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def load_config(self):
        return AutoConfig.from_pretrained(self.model_name)
    
    def get_model_path(self):
        return self.model_path
    
        
        