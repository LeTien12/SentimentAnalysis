from transformers import TrainingArguments , Trainer
import torch

class WeightedTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.1, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
class WeightedTrainer(Trainer):
    def __init__(self, class_weights , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False , num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights).float().to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        p_t = torch.exp(-loss)  # Exp của mất mát CE
        focal_loss = (1 - p_t) ** self.args.gamma * loss * self.args.alpha

        # Trả về kết quả
        return (focal_loss.mean(), outputs) if return_outputs else focal_loss.mean()
        # return (loss, outputs) if return_outputs else loss
     