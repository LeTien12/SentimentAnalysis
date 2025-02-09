import torch
import huggingface_hub
from transformers import AutoTokenizer , AutoModelForSequenceClassification , AutoConfig , pipeline
from .update_model_teacher import WeightedTrainingArguments , WeightedTrainer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import gc
from datasets import DatasetDict


class PretrainModelTeacher:
    def __init__(self,  settings:dict , datasets:DatasetDict):
        
        self.settings = settings
        self.model_path = settings.MODEL_PATH_TEACHER
        self.model_name = settings.MODEL_TEACHER_NAME
        self.huggingface_token = settings.HUGGINGFACE_API
        self.dataset = datasets
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            
        if self.settings.FLAG_TRAINING:
            
            if self.dataset is None:
                raise ValueError("Data path is required to train the model, since the model path does not exist in huggungface hub")
            
            self.tokenizer = self.load_tokenizer()
            self.config = self.load_config()
            self.logging_steps= len(self.dataset['train']) // self.settings.BATCH_SIZE
            self.class_weights = self.load_class_weights(self.dataset['train']['label'])
            self.dataset = self.dataset.map(self.tokenize_text, batched=True)
            self.train_model()
            
        self.model = self.load_model()
        
    def __call__(self , text):
        return self.model(text)
        
    def load_model(self):
        model = pipeline('text-classification' , model = self.model_path , return_all_scores = True)
        return model

    def train_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config).to(self.device)
        
        training_args = WeightedTrainingArguments(
            output_dir = self.model_path,
            learning_rate= self.settings.LEARNING_RATE,
            per_device_train_batch_size=self.settings.BATCH_SIZE,
            per_device_eval_batch_size=self.settings.BATCH_SIZE,
            num_train_epochs=self.settings.EPOCHS,
            weight_decay= self.settings.WEIGHT_DECAY,
            eval_strategy="epoch",
            disable_tqdm=False,
            push_to_hub= self.settings.PUSH_TO_HUB,
            hub_token = self.huggingface_token,
            logging_steps=self.logging_steps,
            log_level="error",
            alpha = self.settings.ALPHA,
            gamma = self.settings.GAMMA,
            report_to="none"
        )
        trainer = WeightedTrainer(class_weights = self.class_weights, model=model, args=training_args,
          compute_metrics=self.compute_metrics,
          train_dataset=self.dataset["train"],
          eval_dataset=self.dataset["validation"],
          tokenizer=self.tokenizer)
        
        trainer.train()
        
        del trainer , model
        gc.collect()
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def tokenize_text(self , batch):
        return self.tokenizer(batch["text"])
    
    def load_config(self):
        class_labels = self.dataset['train'].features['label']
        label2id = {label: idx for idx, label in enumerate(class_labels.names)}
        id2label = {idx: label for idx, label in enumerate(class_labels.names)}
        
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = len(label2id)
        config.id2label = id2label
        config.label2id = label2id
        return config
        
    def load_class_weights(self , train_labels):
        return compute_class_weight(
                class_weight="balanced",
                classes=np.unique(train_labels),  
                y=train_labels
            )
        
        
    