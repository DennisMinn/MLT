import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from base.base_model import BaseModel 

class LitNeuralNet(BaseModel):
    def __init__(self, **config):
        super(LitNeuralNet, self).__init__()
        self.fc1 = nn.Linear(np.prod(config['in_dims']), 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, config['out_dims'])
        
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(config['batch_size'], *config['in_dims'])

    def forward(self, x):
        batch_size, *in_dims = x.shape
        x = x.view(batch_size, -1)
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

        return {'loss': loss, 'logits': logits}

    def validation_step(self, batch, batch_index):
        images, labels = batch
        logits, loss = self.loss(images, labels)
        preds = torch.argmax(logits, 1)
        
        # log metrics
        metrics = self._calculate_metrics(preds, labels)
        metrics['loss'] = loss

        self._log_metrics('validation', metrics)
        
        # log images
        tensorboard = self.logger.experiment
        img_grid = torchvision.utils.make_grid(images)
        tensorboard.add_image('validation/images', img_grid)

        # log logits
        tensorboard.add_histogram('validation/logits', logits)
        return {'loss': loss}
