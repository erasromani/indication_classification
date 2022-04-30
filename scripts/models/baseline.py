import torch

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
    labels = kwargs['labels']
    loss = self.loss_func(logits, labels)
    return {
      "loss": loss,
      "logits": logits,
      "labels": labels
    }


class ClinicalSlidingWindow(ClinicalBioBERT):
  def __init__(self, num_classes, reduction='max', c=None):
    super(ClinicalSlidingWindow, self).__init__(num_classes)
    self.reduction = reduction
    self.c = c
    if reduction == 'mean/max':
      assert c is not None, "c value must be specified for reduction = mean/max"

  
  def forward(self, **kwargs):
    x = self.feature_extractor(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'])
    logits = self.classifier(x['pooler_output'])
    reduced_logits = []
    labels = []
    for _, meta in kwargs['metadata'].groupby('id'):
      indices = meta.index.values
      if self.reduction == 'mean':
        reduced_logits.append(logits[indices].mean(axis=0))
      elif self.reduction == 'max':
        reduced_logits.append(logits[indices].max(axis=0).values)
      elif self.reduction == 'mean/max':
        n = len(indices)
        max_logits = logits[indices].max(axis=0).values
        mean_logits = logits[indices].mean(axis=0)
        reduced_logits.append((max_logits + mean_logits * n / self.c) / (1 + n / self.c))
      else:
        ValueError(f'invalid reduction value {self.reduction} entered')
      label = kwargs['labels'][indices][0]
      labels.append(label)
    labels = torch.stack(labels)
    reduced_logits = torch.stack(reduced_logits)
    loss = self.loss_func(reduced_logits, labels)
    return {
      "loss": loss,
      "logits": reduced_logits,
      "labels": labels
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
    labels = kwargs['labels']
    loss = self.loss_func(logits, labels)
    return {
      "loss": loss,
      "logits": logits,
      "labels": labels
    }
