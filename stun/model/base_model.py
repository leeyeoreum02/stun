from typing import Dict, Union, Tuple, List

import torch
from torch import optim, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class BaseModel(LightningModule):
    def __init__(
        self,
        model,
        criterion,
        learning_rate: float = 5e-4,
        max_epochs: int = 50,
        use_sch: bool = False,
    ) -> None:
        super(BaseModel, self).__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.use_sch = use_sch

        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

        self.labels = []
        self.outputs = []

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Union[Optimizer, _LRScheduler]]]]:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        if not self.use_sch:
            return optimizer

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        image, label = batch
        output = self.model(image)

        loss = self.criterion(output, label)
        psnr_score = self.psnr(output, label)
        ssim_score = self.ssim(output, label)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_psnr', psnr_score, prog_bar=True, logger=True)
        self.log('train_ssim', ssim_score, prog_bar=True, logger=True)

        return {'loss': loss, 'train_psnr': psnr_score, 'train_ssim': ssim_score}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        image, label = batch
        self.labels.extend(label)
        output = self.model(image)
        self.outputs.extend(output)

        loss = self.criterion(output, label)
        psnr_score = self.psnr(output, label)
        ssim_score = self.ssim(output, label)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_score, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_score, prog_bar=True, logger=True)

        return output

    def validation_epoch_end(self, validation_step_outputs: List[Tensor]) -> None:
        all_labels = torch.stack(self.labels)
        all_outputs = torch.stack(self.outputs)

        total_loss = self.criterion(all_outputs, all_labels)
        total_psnr = self.psnr(all_outputs, all_labels)
        total_ssim = self.ssim(all_outputs, all_labels)

        self.log('final_val_loss', total_loss, prog_bar=True, logger=True)
        self.log('final_val_psnr', total_psnr, prog_bar=True, logger=True)
        self.log('final_val_ssim', total_ssim, prog_bar=True, logger=True)

        self.labels = []
        self.outputs = []

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        image = batch
        output = self.model(image)
        return output
