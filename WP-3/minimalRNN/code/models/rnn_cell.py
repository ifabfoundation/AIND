from utils.dependencies import *

class MinimalRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MinimalRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_u = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.__reset_parameters()

    def __reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.W_u.data.uniform_(-stdv, stdv)
        self.U_h.data.uniform_(-stdv, stdv)

        self.bias.data.uniform_(stdv)

    def forward(self, x, h_input):
        u_t = torch.tanh(self.W_x(x))
        f_1 = torch.addmm(self.bias, h_input, self.U_h) # hidden_state * weights + bias
        f_2 = torch.mm(u_t, self.W_u) # new_input * weights
        f_t = torch.sigmoid(f_1 + f_2) # Fprget gate
        h_t = f_t*h_input +(1-f_t)*u_t
        return h_t
