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
        model_path = hf_hub_download(repo_id=self.model_name, 
                                     filename=settings.PATH_FILE_ONNX)
        
        self.session = InferenceSession(model_path, sess_options=session_options)
        self.output_names = [output.name for output in self.session.get_outputs()]

    def __call__(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        
        all_probs = []

        for i in range(0, len(queries), self.setting.BATCH_SIZE):
            batch_queries = queries[i:i + self.setting.BATCH_SIZE]
            inputs = self.tokenizer(batch_queries, return_tensors="np", padding=True, truncation=True)
            logits = self.session.run(self.output_names, dict(inputs))[0]
            probs = softmax(logits, axis=-1)           
            all_probs.append(probs)
        
        all_probs = np.concatenate(all_probs, axis=0)
        num_labels = all_probs.shape[1]  
        label_probs = {self.config.id2label[i]: np.mean(all_probs[:, i]) for i in range(num_labels)}
        return label_probs
    
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def load_config(self):
        return AutoConfig.from_pretrained(self.model_name)
    
    def get_model_path(self):
        return self.model_path
    
        
        