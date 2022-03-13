from typing import Dict, Union, Tuple, List

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

        self.train_psnr = PeakSignalNoiseRatio()
        self.valid_psnr = PeakSignalNoiseRatio()
        self.valid_ssim = StructuralSimilarityIndexMeasure()
        self.train_ssim = StructuralSimilarityIndexMeasure()

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
        psnr_score = self.train_psnr(output, label)
        ssim_score = self.train_ssim(output, label)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_psnr', psnr_score, prog_bar=True, logger=True)
        self.log('train_ssim', ssim_score, prog_bar=True, logger=True)

        return {'loss': loss, 'train_psnr': psnr_score.detach(), 'train_ssim': ssim_score.detach()}

    def training_epoch_end(self, outputs: List[Tensor]) -> None:
        self.train_ssim.reset()

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        image, label = batch
        output = self.model(image)

        loss = self.criterion(output, label)
        psnr_score = self.valid_psnr(output, label)
        ssim_score = self.valid_ssim(output, label)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_score, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_score, prog_bar=True, logger=True)

        return {'loss': loss, 'train_psnr': psnr_score.detach(), 'train_ssim': ssim_score.detach()}

    def validation_epoch_end(self, outputs: List[Tensor]) -> None:
        self.valid_ssim.reset()

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        image = batch
        output = self.model(image)
        return output
