import torch
import numpy as np
import numba
import torch.nn as nn
import torch.nn.functional as F



class MultiLoss(torch.nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()

        # Standard Cross-Entropy Loss (for multi-class classification)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

        # Binary Cross-Entropy Loss (for binary classification)
        self.binary_loss = nn.BCEWithLogitsLoss()


    def forward(self, outputs, batch):
       
        missing_weight , shifted_weight , ic_Direction_weight, bridge_weight, cap_clr_weight = 1, 1, 1, 1, 1

        missing_loss, shifted_loss,  = 0, 0                       # Binary classifiers
        ic_Direction_loss, bridge_loss, cap_clr_loss = 0, 0, 0   # Multi class classifiers

        missing_loss += self.cross_entropy(outputs['missing'], batch['missing'])
        shifted_loss += self.cross_entropy(outputs['shifted'], batch['shifted'])
        ic_Direction_loss += self.cross_entropy(outputs['ic_direction'], batch['ic_direction'])
        bridge_loss += self.cross_entropy(outputs['bridge'], batch['bridge'])
        cap_clr_loss += self.cross_entropy(outputs['capacitor_color'], batch['capacitor_color'])

        loss = missing_loss * missing_weight  + shifted_loss * shifted_weight + \
               ic_Direction_loss * ic_Direction_weight + bridge_loss * bridge_weight + \
               cap_clr_loss * cap_clr_weight
        

        loss_stats = {'loss': loss, 'missing_loss': missing_loss, 'shifted_loss': shifted_loss, 
                  'ic_Direction_loss': ic_Direction_loss, 'bridge_loss': bridge_loss,
                  'cap_clr_loss': cap_clr_loss}
        
        return loss, loss_stats
