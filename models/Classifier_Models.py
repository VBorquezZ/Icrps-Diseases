import torch.nn as nn
import torchvision

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class Classifier_MBN_V0(nn.Module):
  def __init__(self, n_classes, dropout = 0.5, freeze = True):
    super(Classifier_MBN_V0, self).__init__()
    
    self.base = torchvision.models.mobilenet_v3_small(pretrained=True)

    if freeze:
      for param in self.base.parameters():
          param.requires_grad = False

    self.base.avgpool = Identity()
    self.base.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(28224, 1024), 
        nn.ReLU(),
        nn.Dropout(dropout),         
        nn.Linear(1024, n_classes)
    )
    


  def forward(self, input):
    output = self.base(input)
    
    return output
  

