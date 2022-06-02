from models.IRSE import IR_50, IR_SE_50, IR_101, IR_SE_101, IR_152, IR_SE_152

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedLinear(nn.Module):
    """
    Linear layer for classification 
    """
    def __init__(self, in_features=512, out_features=100):
        super(NormalizedLinear, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        x = F.normalize(input)
        W = F.normalize(self.W)
        return F.linear(x, W)

class MainStreamModel(nn.Module):
    def __init__(self, 
                 backbone_name: str = 'irse50', 
                 num_classes: int = 100,
                 input_size: list = [448, 448]):
        super().__init__()
        if backbone_name == 'irse50':
            self.backbone = IR_SE_50(input_size)
        else:
            self.backbone = IR_SE_50(input_size)
        self.fc = NormalizedLinear(in_features=512, out_features=num_classes)

    def forward(self, x):
        emb = self.backbone(x)
        return self.fc(emb)

class DistributedMainStreamModel(nn.Module):

    def __init__(self,
                 device_ids: list = [0,1],
                 backbone_name: str = 'irse50', 
                 num_classes: int = 100,
                 input_size: list = [448, 448]):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MainStreamModel(backbone_name,
                                     num_classes,
                                     input_size)
        self.model = nn.DataParallel(self.model)
        self.model.to(device)

    def forward(self, x):
        return self.model(x)