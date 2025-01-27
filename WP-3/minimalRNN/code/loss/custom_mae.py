from utils.dependencies import nn

class customMAE(nn.Module):
    def __init__(self):
        super(customMAE, self).__init__()
        self.loss = nn.L1Loss(reduction='sum')
        
    def forward(self, tens_pred, tens_true, tens_mask):

        # input all tensors
        tens_ignore = ~ tens_mask
        tens_true[tens_ignore] = 0
        tens_pred[tens_ignore] = 0

        mae = self.loss(tens_pred, tens_true) / tens_mask.sum()

        return mae