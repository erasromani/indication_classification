from torch import nn
from transformers import AutoModel

class ClinicalBERT(nn.Module):
  def __init__(self, num_classes):
      super(ClinicalBERT, self).__init__()
      self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
      self.linear = nn.Linear(768, num_classes)
      self.loss_func = nn.CrossEntropyLoss()
  
  def forward(self, **kwargs):
    x = self.bert(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'])
    logits = self.linear(x['pooler_output'])
    label = kwargs['labels']
    loss = self.loss_func(logits, label)
    return {
      "loss": loss,
      "logits": logits
    }
