from models.baseline import ClinicalBioBERT, ClinicalLongformer


def resolve_model(name, *args, **kwargs):
  if name == 'clinicalbiobert':
    return ClinicalBioBERT(*args, **kwargs)
  elif name == 'clinicallongformer':
    return ClinicalLongformer(*args, **kwargs)
  else:
    ValueError(f"invalid model name {name} entered")