from typing import *

import matplotlib.pyplot as plt
import omegaconf
import pytorch_lightning as pl
import torchmetrics
import scikitplot as skplt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn

from common.specifity import Specificity
from src.common.cf_matrix import make_confusion_matrix, make_auc
from src.common.utils import get_env
from src.pretrained import get_pretrained
from common.utils import read_json, write_json


class BasePLModule(pl.LightningModule):
    def __init__(
        self, conf: Optional[omegaconf.DictConfig] = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.conf: omegaconf.DictConfig = conf
        self.n_classes: int = len(self.conf.labels.class_to_use)
        self.build_metrics()
        self.model: nn.Module = get_pretrained(
            pretrained_name=conf.train.model_type, n_classes=self.n_classes
        )
    
    def build_metrics(self) -> None:
        # Metrics
        self.accuracy: Dict[torchmetrics.Accuracy] = {
            "train": torchmetrics.Accuracy(),
            "val": torchmetrics.Accuracy(),
            "test": torchmetrics.Accuracy(),
        }

        self.roc: Dict[torchmetrics.ROC] = {
            "test": torchmetrics.ROC(num_classes=self.n_classes, compute_on_step=False),
            "val": torchmetrics.ROC(num_classes=self.n_classes, compute_on_step=False),
        }

        self.f1_weighted: Dict[torchmetrics.F1] = {
            "val": torchmetrics.F1(num_classes=self.n_classes, average="macro"),
            "test": torchmetrics.F1(num_classes=self.n_classes, average="macro"),
        }

        self.recall_weighted: Dict[torchmetrics.Recall] = {
            "val": torchmetrics.Recall(
                num_classes=self.n_classes, multiclass=True, average="weighted"
            ),
            "test": torchmetrics.Recall(
                num_classes=self.n_classes, multiclass=True, average="weighted"
            ),
        }

        self.precision_weighted: Dict[torchmetrics.Precision] = {
            "val": torchmetrics.Precision(
                num_classes=self.n_classes, multiclass=True, average="weighted"
            ),
            "test": torchmetrics.Precision(
                num_classes=self.n_classes, multiclass=True, average="weighted"
            ),
        }
        self.specifity_micro: Dict[Specificity] = {
            "val": Specificity(num_classes=self.n_classes, average="micro"),
            "test": Specificity(num_classes=self.n_classes, average="micro"),
        }
        self.specifity_weighted: Dict[Specificity] = {
            "val": Specificity(num_classes=self.n_classes, average="weighted"),
            "test": Specificity(num_classes=self.n_classes, average="weighted"),
        }


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        logits = self.model(x)
        return logits

    def _shared_step(self, x: torch.Tensor, y: torch.Tensor, phase: str):
        logits = self(x)

        loss = F.nll_loss(logits, y)

        y_hat = torch.exp(logits)
        y_preds = torch.argmax(y_hat, dim=-1)
        if phase == "test":
            path_json: str = f"{get_env('PROJECT_ROOT')}/auroc_plot/auc_logits.json"
            auc_logits: dict = read_json(path_json)
            model_name = str(self.conf.train.model_type)
            dt_percentage = str(self.conf.data.subset_percentage)
            if dt_percentage not in auc_logits:
                auc_logits[dt_percentage] = {
                    model_name: {
                    "logits": logits.cpu().tolist(),
                    "gold": y.cpu().tolist()
                }
                }
            else:
                auc_logits[dt_percentage][model_name] = {
                    "logits": logits.cpu().tolist(),
                    "gold": y.cpu().tolist()
                }
            write_json(path_json, auc_logits)
            
            self.roc[phase](logits.cpu(), y.cpu())
        self.accuracy[phase](y_preds.cpu(), y.cpu())
        self.f1_weighted[phase](y_preds.cpu(), y.cpu())
        self.recall_weighted[phase](y_preds.cpu(), y.cpu())
        self.precision_weighted[phase](y_preds.cpu(), y.cpu())
        self.specifity_micro[phase](y_preds.cpu(), y.cpu())
        self.specifity_weighted[phase](y_preds.cpu(), y.cpu())

        return {
            f"{phase}_loss": loss,
            f"{phase}_preds": y_preds.cpu().tolist(),
            f"{phase}_y": y.cpu().tolist(),
        }

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        # during the test phase we need to compute just the loss
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        y_hat = torch.exp(logits)
        y_preds = torch.argmax(y_hat, dim=-1)
        self.accuracy["train"](y_preds.cpu(), y.cpu())
        return {"loss": loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        # shared function called by all epoch_step
        x, y = batch
        return self._shared_step(x, y, phase="val")

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        # shared function called by all epoch_step
        x, y = batch
        return self._shared_step(x, y, phase="test")

    def _shared_epoch_end(self, outputs: List[dict], phase: str) -> None:
        # shared function called by all epoch_end

        if phase == "test":
            epoch_y_preds: List[int] = sum(
                [x[f"{phase}_preds"] for x in outputs], []
            )  # flattening of list
            epoch_y: List[int] = sum(
                [x[f"{phase}_y"] for x in outputs], []
            )  # flattening of list
            categories = ["B", "C", "N", "V"]
            cf_matrix = confusion_matrix(
                y_pred=epoch_y_preds,
                y_true=epoch_y,
            )
            print(cf_matrix)
            """
            make_confusion_matrix(
                cf_matrix,
                categories=categories,
                figsize=(7, 7),
                sum_stats=False,
                count=False,
                percent=True,
                cbar=False,
                title=self.conf.train.model_type,
            )
            plt.savefig(f"{get_env('PROJECT_ROOT')}/confusion_matrix/{self.conf.train.model_type}_title.png")
            """

            fpr, tpr, _thresholds = self.roc[phase].compute()

            auc_output_dict: dict = make_auc(fpr, tpr)

            auc_img_path = f"{get_env('PROJECT_ROOT')}/auroc_plot/{self.conf.train.model_type}_no_title_auc_{self.conf.data.subset_percentage}.png"
            plt.savefig(auc_img_path)

            path_json: str = f"{get_env('PROJECT_ROOT')}/auroc_plot/auc_dict.json"
            auc_dict: dict = read_json(path_json)
            if str(self.conf.data.subset_percentage) not in auc_dict:
                auc_dict[str(self.conf.data.subset_percentage)] = {
                    str(self.conf.train.model_type): auc_output_dict
                }
            else:
                auc_dict[str(self.conf.data.subset_percentage)][str(self.conf.train.model_type)] = auc_output_dict
            write_json(path_json, auc_dict)

        accuracy = self.accuracy[phase].compute()
        f1_weighted = self.f1_weighted[phase].compute()
        precison_weighted = self.precision_weighted[phase].compute()
        recall_weighted = self.recall_weighted[phase].compute()
        specifity_micro = self.specifity_micro[phase].compute()
        specifity_weighted = self.specifity_weighted[phase].compute()

        self.log(f"{phase}_accuracy", accuracy, prog_bar=True)
        self.log(f"{phase}_f1_weighted", f1_weighted, prog_bar=True)
        self.log(f"{phase}_precision_weighted", precison_weighted, prog_bar=True)
        self.log(f"{phase}_recall_weighted", recall_weighted, prog_bar=True)
        self.log(f"{phase}_specifity_micro", specifity_micro, prog_bar=True)
        self.log(f"{phase}_specifity_weighted", specifity_weighted, prog_bar=True)

    def training_epoch_end(self, outputs) -> None:
        accuracy = self.accuracy["train"].compute()
        self.log(f"train_accuracy", accuracy, prog_bar=True)

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, phase="val")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, phase="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.conf.model.learning_rate)
