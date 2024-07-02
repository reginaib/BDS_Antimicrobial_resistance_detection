import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import (MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
                                        MultilabelAccuracy, MultilabelRecall, MultilabelPrecision, MultilabelF1Score)
from torcheval.metrics import MultilabelAUPRC, MulticlassAUPRC
from torchmetrics import MetricCollection


class MALDIMLP(pl.LightningModule):
    def __init__(self, hidden_dim, n_hidden_layers, lr, dr,
                 input_dim, species_classes=3, resistance_labels=5):
        super().__init__()
        self.lr = lr
        # Store parameters
        self.save_hyperparameters()
        
        # Initial layer
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dr)]
        
        # Additional hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dr)])
        
        # Sequential model for the base encoder
        self.encoder = nn.Sequential(*layers)
        
        # Species classification head
        self.species_head = nn.Linear(hidden_dim, species_classes)
        
        # Resistance prediction head
        self.resistance_head = nn.Linear(hidden_dim, resistance_labels)
        
        # Metrics for species classification (multiclass)
        self.metrics_species_train = MetricCollection([MulticlassAccuracy(num_classes=species_classes,  average='macro'),
                                                       MulticlassPrecision(num_classes=species_classes, average='macro'),
                                                       MulticlassRecall(num_classes=species_classes, average='macro'),
                                                       MulticlassF1Score(num_classes=species_classes, average='macro')],
                                                       prefix='Train_species/')
        self.metrics_species_train_prc = MulticlassAUPRC(num_classes=species_classes, average='macro')

        self.metrics_species_valid = MetricCollection([MulticlassAccuracy(num_classes=species_classes, average='macro'),
                                                       MulticlassPrecision(num_classes=species_classes, average='macro'),
                                                       MulticlassRecall(num_classes=species_classes, average='macro'),
                                                       MulticlassF1Score(num_classes=species_classes, average='macro')],
                                                       prefix='Valid_species/')
        self.metrics_species_valid_prc = MulticlassAUPRC(num_classes=species_classes, average='macro')
        
        self.metrics_species_test = MetricCollection([MulticlassAccuracy(num_classes=species_classes, average='macro'),
                                                      MulticlassPrecision(num_classes=species_classes, average='macro'),
                                                      MulticlassRecall(num_classes=species_classes, average='macro'),
                                                      MulticlassF1Score(num_classes=species_classes, average='macro')],
                                                      prefix='Test_species/')
        self.metrics_species_test_prc = MulticlassAUPRC(num_classes=species_classes, average='macro')
        

        # Metrics for resistance prediction (multilabel)
        self.metrics_resistance_train = MetricCollection([MultilabelAccuracy(num_labels=resistance_labels, average=None),
                                                        MultilabelRecall(num_labels=resistance_labels, average=None),
                                                        MultilabelPrecision(num_labels=resistance_labels, average=None),
                                                        MultilabelF1Score(num_labels=resistance_labels, average=None)], 
                                                        prefix='Train_resistance/')
        self.metrics_resistance_train_prc = MultilabelAUPRC(num_labels=resistance_labels, average=None)

        self.metrics_resistance_valid = MetricCollection([MultilabelAccuracy(num_labels=resistance_labels, average=None),
                                                          MultilabelRecall(num_labels=resistance_labels, average=None),
                                                          MultilabelPrecision(num_labels=resistance_labels, average=None),
                                                          MultilabelF1Score(num_labels=resistance_labels, average=None)],
                                                          prefix='Valid_resistance/')
        self.metrics_resistance_valid_prc = MultilabelAUPRC(num_labels=resistance_labels, average=None)
        
        self.metrics_resistance_test = MetricCollection([MultilabelAccuracy(num_labels=resistance_labels, average=None),
                                                         MultilabelRecall(num_labels=resistance_labels, average=None),
                                                         MultilabelPrecision(num_labels=resistance_labels, average=None),
                                                         MultilabelF1Score(num_labels=resistance_labels, average=None)],
                                                         prefix='Test_resistance/')
        self.metrics_resistance_test_prc = MultilabelAUPRC(num_labels=resistance_labels, average=None)
        self.save_hyperparameters()

    
    def forward(self, x):
        # Pass input through the encoder layers
        x = self.encoder(x)

        # Predict species labels using the species head
        species_pred = self.species_head(x)

        # Predict resistance labels using the resistance head
        resistance_pred = self.resistance_head(x)
        
        # Return both species and resistance predictions
        return species_pred, resistance_pred

    
    def training_step(self, batch, batch_idx):
        # Extract data from the batch
        x, species, resistance = batch
        
        # Forward pass through the model
        species_pred, resistance_pred = self(x)
        
        # Calculate species classification loss using cross-entropy
        species_loss = F.cross_entropy(species_pred, species)

        # Create a mask to handle missing values in resistance labels
        mask = ~resistance.isnan()

        # Calculate resistance prediction loss using binary cross-entropy and mask
        resistance_loss = F.binary_cross_entropy_with_logits(resistance_pred * mask, resistance.nan_to_num())
        
        # Total loss
        loss = species_loss + resistance_loss

        # Update metrics for species classification
        self.metrics_species_train.update(species_pred.softmax(-1), species)
        self.metrics_species_train_prc.update(species_pred.softmax(-1), species)

        # Update metrics for resistance prediction
        self.metrics_resistance_train.update(resistance_pred.sigmoid() * mask, resistance.nan_to_num())
        self.metrics_resistance_train_prc.update(resistance_pred.sigmoid() * mask, resistance.nan_to_num()) 

        # Log and return the total loss
        self.log('Training/train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, species, resistance = batch
        species_pred, resistance_pred = self(x)
        
        species_loss = F.cross_entropy(species_pred, species)
        mask = ~resistance.isnan()
        resistance_loss = F.binary_cross_entropy_with_logits(resistance_pred * mask, resistance.nan_to_num())
        loss = species_loss + resistance_loss

        self.metrics_species_valid.update(species_pred.softmax(-1), species)
        self.metrics_species_valid_prc.update(species_pred.softmax(-1), species)
        self.metrics_resistance_valid.update(resistance_pred.sigmoid() * mask, resistance.nan_to_num())
        self.metrics_resistance_valid_prc.update(resistance_pred.sigmoid() * mask, resistance.nan_to_num()) 

        self.log('Validation/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, species, resistance = batch
        species_pred, resistance_pred = self(x)
        
        species_loss = F.cross_entropy(species_pred, species)
        mask = ~resistance.isnan()
        resistance_loss = F.binary_cross_entropy_with_logits(resistance_pred * mask, resistance.nan_to_num())
        loss = species_loss + resistance_loss

        self.metrics_species_test.update(species_pred.softmax(-1), species)
        self.metrics_species_test_prc.update(species_pred.softmax(-1), species)
        self.metrics_resistance_test.update(resistance_pred.sigmoid() * mask, resistance.nan_to_num())
        self.metrics_resistance_test_prc.update(resistance_pred.sigmoid() * mask, resistance.nan_to_num()) 

        self.log('Test/test_loss', loss)

    
    def on_train_epoch_end(self):
        # Log metrics for species classification on training set
        self.log_dict(self.metrics_species_train.compute())
        self.log('Train_species/MultiClassAUPRC', self.metrics_species_train_prc.compute())
        
        # Reset metrics for species classification on training set
        self.metrics_species_train.reset()
        self.metrics_species_train_prc.reset()
    
        # Log metrics for resistance prediction on training set
        # Fixed log for multiple labels
        self.log_dict({f'{k}_{n}': x for k, v in self.metrics_resistance_train.compute().items() for n, x in enumerate(v)})
        self.log_dict({f'Train_resistance/MultiLabelAUPRC_{n}': x for n, x in enumerate(self.metrics_resistance_train_prc.compute())})
        
        # Reset metrics for resistance prediction on training set
        self.metrics_resistance_train.reset()
        self.metrics_resistance_train_prc.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics_species_valid.compute())
        self.log('Valid_species/MultiClassAUPRC', self.metrics_species_valid_prc.compute())
        self.metrics_species_valid.reset()
        self.metrics_species_valid_prc.reset() 
    
        self.log_dict({f'{k}_{n}': x for k, v in self.metrics_resistance_valid.compute().items() for n, x in enumerate(v)})
        self.log_dict({f'Valid_resistance/MultiLabelAUPRC_{n}': x for n, x in enumerate(self.metrics_resistance_valid_prc.compute())})

        self.metrics_resistance_valid.reset()
        self.metrics_resistance_valid_prc.reset()
        
    def on_test_epoch_end(self):
        self.log_dict(self.metrics_species_test.compute())
        self.log('Test_species/MultiClassAUPRC', self.metrics_species_test_prc.compute())
        self.metrics_species_test.reset()
        self.metrics_species_test_prc.reset()
    
        self.log_dict({f'{k}_{n}': x for k, v in self.metrics_resistance_test.compute().items() for n, x in enumerate(v)})
        self.log_dict({f'Test_resistance/MultiLabelAUPRC_{n}': x for n, x in enumerate(self.metrics_resistance_test_prc.compute())})
        
        self.metrics_resistance_test.reset()
        self.metrics_resistance_test_prc.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)