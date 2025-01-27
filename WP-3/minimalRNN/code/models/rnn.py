from utils.dependencies import *
from rnn_cell import MinimalRNNCell

class MinimalRNN(nn.Module):

    def __init__(self, s_size, g_size, static_size, hidden_size, h_drop, x_drop, n_layers, cuda_bool):
        super(MinimalRNN, self).__init__()
        self.h_drop = 1 - h_drop # probability of maintenance
        self.x_drop = 1 - x_drop # probability of maintenance
        self.s_size = s_size
        self.g_size = g_size
        self.hidden_size = hidden_size

        self.W_s = nn.Linear(hidden_size, s_size)
        self.W_g = nn.Linear(hidden_size, g_size)
        self.cells = nn.ModuleList()
        self.cells.append(MinimalRNNCell(input_size=g_size+s_size+static_size, hidden_size=hidden_size))
        for _ in range(1, n_layers):
            self.cells.append(MinimalRNNCell(hidden_size, hidden_size))
        self.gpu = cuda_bool
        

    def __init_hstate(self, batch_size):
        h_states = []
        for cell in self.cells:
            tensor = torch.zeros(batch_size, cell.hidden_size)
            if self.gpu:
                tensor = tensor.cuda()
            h_states.append(tensor)
        return h_states
    
    def __dropout_mask(self, batch_size):
        h_mask = []
        for cell in self.cells:
            tensor = torch.ones(batch_size, cell.hidden_size)
            if self.gpu:
                tensor = tensor.cuda()
            h_mask.append(tensor)

        x_mask = torch.ones(batch_size, self.g_size)
        if self.gpu:
            x_mask = x_mask.cuda()

        if self.training:
            x_mask.bernoulli_(self.x_drop)
            for mask in h_mask:
                mask.bernoulli_(self.h_drop)

        return x_mask, h_mask
    
    def predict(self, s_input, g_input, static_input, hid, masks):
        #s = cathegorical data
        #g = contineous data
        mask_i, mask_h = masks
        g_input_drop = g_input * mask_i      #dropping out just the contineous part of th inputs
        x_input = torch.cat([s_input, g_input_drop, static_input], dim = -1)

        next_hid = []
        h_t = x_input
        for cel, prev_hid, mask in zip(self.cells, hid, mask_h):
            h_t = cel(h_t, prev_hid * mask)
            next_hid.append(h_t)

        #s_pred = nn.functional.softmax(self.W_s(h_t), dim=-1)
        s_pred = self.W_s(h_t)
        g_pred = self.W_g(h_t)+g_input
        return s_pred, g_pred, next_hid
    
    def forward (self, _s_seq, _g_seq, static_seq):         #input in forma di tensori
        pred_s_seq, pred_g_seq = [], []
        
        # Avoid changes on original data
        s_seq = _s_seq.clone()
        g_seq = _g_seq.clone()

        batch_size = _g_seq.shape[0]            # perche selezionano il secondo valore dello shape? che struttura hanno i dati?

        hidden_state = self.__init_hstate(batch_size)
        masks = self.__dropout_mask(batch_size)

        for i,j in zip(range(g_seq.shape[1]), range(1, g_seq.shape[1])):
            s_pred, g_pred, hidden_state = self.predict(s_seq[:, i, :], g_seq[:, i, :], static_seq[:, i, :], hidden_state, masks)
            pred_s_seq.append(s_pred)
            pred_g_seq.append(g_pred)

            # During training, use real values. In eval mode, use predicted values.
            if self.training:
                # Fill in the missing features of the next timepoint
                idx = torch.isnan(s_seq[:, j, :])
                s_seq[:, j, :].masked_scatter_(idx, s_pred[idx])     

                idx = torch.isnan(g_seq[:, j, :])
                g_seq[:, j, :].masked_scatter_(idx, g_pred[idx])
            else:
                s_seq[:, j, :] = s_pred
                g_seq[:, j, :] = g_pred

        # Returns results in a single tensor with an additional dimension for the timestamp
        return torch.stack(pred_s_seq), torch.stack(pred_g_seq)


