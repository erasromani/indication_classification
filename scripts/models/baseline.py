from torch import nn
from transformers import AutoModel


def resolve_feature_extractor(name):
  if name == "clinicalbiobert":
    return AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
  elif name == "clinicallongformer":
    return AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
  else:
      ValueError(f"invalid feature extractor name {name}")


class ClinicalBioBERT(nn.Module):
  def __init__(self, num_classes):
      super(ClinicalBioBERT, self).__init__()
      self.feature_extractor = resolve_feature_extractor("clinicalbiobert")
      self.classifier = nn.Linear(self.feature_extractor.get_input_embeddings().embedding_dim, num_classes)
      self.loss_func = nn.CrossEntropyLoss()
  
  def forward(self, **kwargs):
    x = self.feature_extractor(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'])
    logits = self.classifier(x['pooler_output'])
    label = kwargs['labels']
    print(logits.shape, label.shape)
    loss = self.loss_func(logits, label)
    return {
      "loss": loss,
      "logits": logits
    }

class ClinicalLongformer(nn.Module):
  def __init__(self, num_classes):
      super(ClinicalLongformer, self).__init__()
      self.feature_extractor = resolve_feature_extractor("clinicallongformer")
      self.classifier = nn.Linear(self.feature_extractor.get_input_embeddings().embedding_dim, num_classes)
      self.loss_func = nn.CrossEntropyLoss()
  
  def forward(self, **kwargs):
    x = self.feature_extractor(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'])
    logits = self.classifier(x['pooler_output'])
    label = kwargs['labels']
    loss = self.loss_func(logits, label)
    return {
      "loss": loss,
      "logits": logits
    }