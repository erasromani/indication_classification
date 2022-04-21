import itertools
import os
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from sklearn.metrics import f1_score
from functools import partial
from utils import epoch_iter, save_weights

@torch.no_grad()
def eval_loop(model, dataloader, device):
    """Run validation phase."""
    model.eval()

    # Keeping track of metrics
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    all_labels = []
    all_preds = []

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}        
        outputs = model(**batch)
        loss = outputs["loss"]

        # Only count non-padding tokens
        # (Same idea as ignore_index=PAD_IDX above)
        preds = outputs['logits'].argmax(-1)
        labels = batch['labels']
        correct_preds = (labels == preds).sum()
        all_labels.append(labels)
        all_preds.append(preds)

        # Keeping track of metrics
        total_loss += loss.item()
        total_correct += correct_preds.item()
        total_count += preds.shape[0]
    all_labels = torch.cat(all_labels).cpu()
    all_preds = torch.cat(all_preds).cpu()
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "f1_score": f1_score(all_labels, all_preds, average='macro')
    }

def train_step(optimizer, model, batch, clip_grad_norm=None):
    """Run a single train step."""
    model.train()
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    optimizer.step()
    return loss.item()


def lr_lambda(current_step: int, warmup_steps: int, total_steps: int, decay_type='linear'):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    if decay_type is None:
        return 1.0
    elif decay_type == 'linear':
        w = - 1 / (total_steps - warmup_steps)
        return (current_step - warmup_steps) * w + 1.0
    elif decay_type == 'cosine':
        w = np.pi / (total_steps - warmup_steps)
        return 0.5 * np.cos(w * (current_step - warmup_steps)) + 0.5
    else:
        raise ValueError('invalid decay_type {} entered'.format(decay_type))


# TODO: Add gradient clipping
def train_loop(
    train_dataloader, 
    val_dataloader, 
    model, optimizer, 
    num_epochs, 
    warmup_steps, 
    device, 
    save_path, 
    logging_steps=5, 
    checkpoint_metric='f1_score',
    clip_grad_norm=None
    ):

  total_steps = num_epochs * len(train_dataloader)
  lr_scheduler = None if warmup_steps is None \
                      else torch.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, warmup_steps=warmup_steps, 
                                                                                          total_steps=total_steps))
  train_loss_list = []
  writer = SummaryWriter(log_dir=os.path.join(save_path, 'tb_logs'))
  model.to(device)
  max_metric = - float('inf')
  for step, epoch, batch in zip(range(total_steps), epoch_iter(num_epochs, train_dataloader), itertools.cycle(train_dataloader)):
      batch = {key: value.to(device) for key, value in batch.items()}
      loss_val = train_step(
          optimizer=optimizer,
          model=model,
          batch=batch,
          clip_grad_norm=clip_grad_norm,
      )
      writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], step)
      if lr_scheduler is not None:
          lr_scheduler.step()

      writer.add_scalar("epoch", epoch, step)
      writer.add_scalar("loss/train", loss_val, step)
      train_loss_list.append(loss_val)
      if step % logging_steps == 0 and step != 0:
            val_results = eval_loop(
                model=model,
                dataloader=val_dataloader,
                device=device
            )
            for key, value in val_results.items():
                writer.add_scalar(f"{key}/val", value, step)
            print("Step: {}/{}, val acc: {:.3f}, val f1: {:.3f}".format(
                step, 
                total_steps,
                val_results["accuracy"],
                val_results["f1_score"])
            )
            if max_metric < val_results[checkpoint_metric]:
                max_metric = val_results[checkpoint_metric]
                save_weights(model, optimizer, f'{save_path}/model_best_val.pt', epoch=epoch, step=step, checkpoint_metric=checkpoint_metric, metric_value=max_metric)

  save_weights(model, optimizer, f'{save_path}/model_last_epoch.pt', epoch=epoch, step=step)
  writer.flush()
  writer.close()