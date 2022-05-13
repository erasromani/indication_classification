from models.baseline import ClinicalBioBERT, ClinicalLongformer, ClinicalSlidingWindow


def resolve_model(name, *args, **kwargs):
  if name == 'clinicalbiobert':
    return ClinicalBioBERT(*args)
  elif name == 'clinicallongformer':
    return ClinicalLongformer(*args)
  elif name == "clinicalslidingwindow":
      return ClinicalSlidingWindow(*args, **kwargs)
  else:
    raise ValueError(f"invalid model name {name} entered")