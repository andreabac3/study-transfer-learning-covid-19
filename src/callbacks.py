from typing import Dict, List

import pytorch_lightning as pl
import torch

import wandb


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, class_to_use_list: List[str], num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
        # Convert the class_to_use_list into a dictionary

        self.idx_to_class: Dict[int, str] = {i: class_to_use_list[i] for i in range(0, len(class_to_use_list))}

    def on_test_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        trainer.logger.experiment.log(
            {
                f"Test dataset": [  # TODO add batch-0 and class_used
                    wandb.Image(
                        x,
                        caption=f"Pred:{self.idx_to_class[pred.item()]}, Label:{self.idx_to_class[y.item()]}",
                    )
                    for x, pred, y in zip(val_imgs, preds, self.val_labels)
                ],
                "global_step": trainer.global_step,
            }
        )
