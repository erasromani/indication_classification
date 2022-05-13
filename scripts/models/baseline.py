import torch
import torch.nn.functional as F
import collections

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
    if reduction == 'mean/max':
      self.c = c
      assert c is not None, "c value must be specified for reduction = mean/max"
    elif reduction == 'attention':
      self.attention = AttentionModule(self.feature_extractor.get_input_embeddings().embedding_dim, 128)

  
  def forward(self, **kwargs):
    x = self.feature_extractor(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'])
    if self.reduction != 'attention':
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
      elif self.reduction == 'attention':
        reduced_x = self.attention(x['pooler_output'][indices])
        reduced_logits.append(self.classifier(reduced_x).squeeze(0))
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
  def __init__(self, num_classes, pretrain_path=None):
    super(ClinicalLongformer, self).__init__()
    self.feature_extractor = resolve_feature_extractor("clinicallongformer")
    self.classifier = nn.Linear(self.feature_extractor.get_input_embeddings().embedding_dim, num_classes)
    self.loss_func = nn.CrossEntropyLoss()

    if pretrain_path is not None:
      checkpoint = torch.load(pretrain_path)
      new_state_dict = collections.OrderedDict()
      for k, v in checkpoint.items():
        name = k.replace('longformer.', '')
        new_state_dict[name] = v
      self.feature_extractor.load_state_dict(new_state_dict, strict=False)

  def forward(self, inference=False, **kwargs):
    x = self.feature_extractor(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'])
    logits = self.classifier(x['pooler_output'])
    if inference:
      return logits
    labels = kwargs['labels']
    loss = self.loss_func(logits, labels)
    return {
      "loss": loss,
      "logits": logits,
      "labels": labels
    }


class AttentionModule(nn.Module):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf
    """

    def __init__(self, in_dim, hidden_dim):
        super(AttentionModule, self).__init__()
        # The gated attention mechanism
        self.mil_attn_V = nn.Linear(in_dim, hidden_dim, bias=False)
        self.mil_attn_U = nn.Linear(in_dim, hidden_dim, bias=False)
        self.mil_attn_w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        """
        Function that takes in the hidden representations of multiple documents and use attention to generate a single hidden vector
        :param x:
        :return:
        """
        # calculate the attn score
        attn_projection = torch.sigmoid(self.mil_attn_U(x)) * \
                          torch.tanh(self.mil_attn_V(x))
        attn_score = self.mil_attn_w(attn_projection)
        # use softmax to map score to attention
        attn = F.softmax(attn_score, dim=0)

        # final hidden vector
        reduced_x = (attn * x).sum(dim=0, keepdim=True)

        return reduced_x