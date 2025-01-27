from utils.dependencies import nn, torch

class customCrossEntropy(nn.Module):
    def __init__(self):
        super(customCrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, tens_pred, tens_true, tens_mask):

        batch_size = tens_true.shape[0]
        n_classes = tens_pred.shape[-1]

        tens_true_r = tens_true.reshape(-1, 1) # (10,21) -> (10*21,1) -> (n_pati * n_visits, 1)
        tens_pred_r = tens_pred.reshape(-1, n_classes) # (10,21,3) -> (10*21,3)
        tens_mask_r = tens_mask.reshape(-1, 1) # (10,21) -> (10*21,1) -> (n_pati * n_visits, 1)

        # input all tensors
        tens_true_r = tens_true_r[tens_mask_r]
        tens_true_r = tens_true_r.type(torch.LongTensor).cuda()
        tens_pred_r = tens_pred_r[tens_mask_r.squeeze(-1)]

        cross_entropy = self.loss(tens_pred_r, tens_true_r) / tens_true_r.shape[0]

        return cross_entropy