import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from base.base_model import BaseModel 

import wandb
class LitConvNet(BaseModel):
    def __init__(self, **config):
        super(LitConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config['out_dims'])
        
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(config['batch_size'], *config['in_dims'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def loss(self, images, labels):
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        logits = F.log_softmax(outputs, dim=1)
        return logits, loss
    
    def training_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)
        preds = torch.argmax(logits, 1)
        
        # log metrics
        metrics = self._calculate_metrics(preds, labels)
        metrics['loss'] = loss

        self._log_metrics('train', metrics)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)
        preds = torch.argmax(logits, 1)
        
        # log metrics
        metrics = self._calculate_metrics(preds, labels)
        metrics['loss'] = loss

        self._log_metrics('validation', metrics)

        wandb_logger = self.logger.experiment

        # log images
        # img_grid = torchvision.utils.make_grid(images)
        # wandb_logger.log({'validation/imgs': wandb.Image(img_grid)}) 

        # log logits
        wandb_logger.log({'validation/logits': wandb.Histogram(logits)})
        return {'loss': loss}
