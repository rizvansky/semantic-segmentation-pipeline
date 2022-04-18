import torch
import torch.nn as nn
import uuid
import os
from pipe.registry import registry
from tqdm import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from typing import Union, Optional, List


@registry.register
class BaseSemanticSegmentationPipeline:
    def __init__(
            self,
            device: Union[torch.device, str],
            model: nn.Module,
            num_epochs: int,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            optimizer: Optimizer,
            criterion: nn.Module,
            logs_dir: Optional[str] = "./logs",
            logs_name: Optional[str] = ""
    ):
        self.device = device
        self.model = model
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.summary = None
        if logs_dir is not None:
            if not os.path.isdir(logs_dir):
                os.makedirs(logs_dir)
            if logs_name is None or os.path.isdir(os.path.join(logs_dir, logs_name)):
                logs_name = uuid.uuid4()

            self.summary = SummaryWriter(os.path.join(logs_dir, logs_name))

    def write_logs(self, metrics):
        if self.summary:
            pass

    def run_train_stage(self, epoch: int):
        progress_bar = tqdm(total=len(self.train_dataloader), position=epoch, desc=f"Epoch {epoch}. Training...")
        self.model.train()
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            progress_bar.update(1)

    def run_val_stage(self, epoch: int):
        progress_bar = tqdm(total=len(self.val_dataloader), position=epoch, desc=f"Epoch {epoch}. Validating...")
        self.model.eval()
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
                loss = self.criterion(outputs, y)


            progress_bar.update(1)

    def run(self):
        for epoch in range(self.num_epochs):
            self.run_train_stage(epoch)
            self.run_val_stage(epoch)
