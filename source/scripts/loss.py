import torch.nn as nn
from torchmetrics.classification import Dice

class DiceEntropyLoss(nn.Module):
    def __init__(self, device, classes, weight=None, size_average=True, alpha=0.5):
        super(DiceEntropyLoss, self).__init__()
        self.CEL = nn.CrossEntropyLoss()
        self.dice = Dice(average='macro', num_classes=classes+1).to(device)
        self.alpha = alpha
    def forward(self,preds, targets):
        cel = self.CEL(preds,targets)
        dice = self.dice(preds,targets)
        return self.alpha*(cel) + (1-self.alpha)*dice