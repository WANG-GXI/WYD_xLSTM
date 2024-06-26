import torch
import torch.nn as nn


class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters for input, forget, and output gates
        self.w_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_z = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.r_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, 1)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_i)
        nn.init.xavier_uniform_(self.w_f)
        nn.init.xavier_uniform_(self.w_o)
        nn.init.xavier_uniform_(self.w_z)

        nn.init.orthogonal_(self.r_i)
        nn.init.orthogonal_(self.r_f)
        nn.init.orthogonal_(self.r_o)
        nn.init.orthogonal_(self.r_z)

        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_z)

    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states
        # input gate
        i_tilda = torch.matmul(x, self.w_i) + torch.matmul(h_prev, self.r_i) + self.b_i
        i_t = torch.exp(i_tilda)

        # forget gate
        f_tilda = torch.matmul(x, self.w_f) + torch.matmul(h_prev, self.r_f) + self.b_f

        # 论文提到选择用sigmoid或者exp 实验发现用sigmoid效果极佳
        # f_t = torch.exp(f_tilda)
        f_t = self.sigmoid(f_tilda)

        # output gate
        o_tilda = torch.matmul(x, self.w_o) + torch.matmul(h_prev, self.r_o) + self.b_o
        o_t = self.sigmoid(o_tilda)

        # 激活函数层
        z_tilda = torch.matmul(x, self.w_z) + torch.matmul(h_prev, self.r_z) + self.b_z
        z_t = torch.tanh(z_tilda)

        # Stabilizer state update 作者新加的稳定门
        # 这里不加log应该也可以判断大小 后期看看速度有没有影响【2024-5-29】
        m_t = torch.max(torch.log(f_t) + m_prev, torch.log(i_t))

        # Stabilized gates
        i_prime = torch.exp(i_tilda - m_t)
        f_prime = torch.exp(torch.log(f_t) + m_prev - m_t)

        # 这里用f_prime参考论文公式17下面的解释
        c_t = f_prime * c_prev + i_prime * z_t
        n_t = f_prime * n_prev + i_prime

        # hidden state
        h_tilda = c_t / n_t
        h_t = o_t * h_tilda

        # 为了保持输出为1维 加一个全连接层
        out = self.fc(h_t)  # 取最后一个时间步的输出

        return out, (h_t, c_t, n_t, m_t)


class WYDsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(WYDsLSTM, self).__init__()
        self.layers = nn.ModuleList(
            [
                sLSTMCell(
                    input_size if i == 0 else hidden_size, hidden_size
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, initial_states=None):
        """
        在这里定义x的输入格式是(batch, sequence, feature)  等价于nn.LSTM中batch_first=True的效果
        """
        batch_size, seq_len, _ = x.size()
        if initial_states is None:
            initial_states = [
                (
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                )
                for _ in self.layers
            ]

        outputs = []
        current_states = initial_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                h_t, new_state = layer(x_t, state)
                new_states.append(new_state)
                x_t = h_t  # Pass the output to the next layer
            outputs.append(h_t.unsqueeze(1))
            current_states = new_states

        outputs = torch.cat(
            outputs, dim=1
        )  # Concatenate on the time dimension
        return outputs, current_states


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters for input, forget, and output gates
        self.w_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_o = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.w_q = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_k = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_v = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.b_q = nn.Parameter(torch.Tensor(hidden_size))
        self.b_k = nn.Parameter(torch.Tensor(hidden_size))
        self.b_v = nn.Parameter(torch.Tensor(hidden_size))

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, 1)
        self.group_norm = nn.GroupNorm(1, hidden_size)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_i)
        nn.init.xavier_uniform_(self.w_f)
        nn.init.xavier_uniform_(self.w_o)

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_o)

        nn.init.zeros_(self.b_q)
        nn.init.zeros_(self.b_k)
        nn.init.zeros_(self.b_v)

    def forward(self, x, states):
        # 在mlstm中h_prev没有用到
        h_prev, c_prev, n_prev, m_prev = states

        #  Calculate the input, forget, output, query, key and value gates
        i_tilda = torch.matmul(x, self.w_i) + self.b_i
        i_t = torch.exp(i_tilda)

        f_tilda = torch.matmul(x, self.w_f) + self.b_f
        # 论文提到选择用sigmoid或者exp 实验发现用sigmoid效果极佳
        f_t = torch.exp(f_tilda)
        # f_t = self.sigmoid(f_tilda)

        o_tilda = torch.matmul(x, self.w_o) + self.b_o
        o_t = self.sigmoid(o_tilda)

        q_t = torch.matmul(x, self.w_q) + self.b_q

        k_t = torch.matmul(x, self.w_k) / (self.hidden_size ** 0.5) + self.b_k   #torch.sqrt(torch.tensor(self.hidden_size)) + self.b_k

        v_t = torch.matmul(x, self.w_v) + self.b_v

        # Stabilization state
        m_t = torch.max(torch.log(f_t) + m_prev, torch.log(i_t))
        i_prime = torch.exp(i_tilda - m_t)
        f_prime = torch.exp(torch.log(f_t) + m_prev - m_t)

        # 这里用f_prime参考论文公式19 20下面的解释
        c_t = f_prime * c_prev + i_prime * (v_t * k_t)
        n_t = f_prime * n_prev + i_prime * k_t

        h_tilda = (c_t * q_t) / (torch.max(torch.abs(n_t.T @ q_t), 1)[0])  # o * (c @ q) / max{|n.T @ q|, 1}
        h_t = o_t * h_tilda

        # 为了保持输出为1维 加一个全连接层
        out = self.fc(h_t)  # 取最后一个时间步的输出

        return out, (h_t, c_t, n_t, m_t)


class WYDmLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(WYDmLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                mLSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )

    def forward(self, x, state=None):
        """
        在这里定义x的输入格式是(batch, sequence, feature)  等价于nn.LSTM中batch_first=True的效果
        """
        assert x.ndim == 3
        batch_size, seq_len, _ = x.size()
        if state is None:
            state = [
                (
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ),
                )
                for _ in self.layers
            ]

        outputs = []
        current_states = state

        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, Cell in zip(self.layers, current_states):
                h_t, new_state = layer(x_t, Cell)
                new_states.append(new_state)
                x_t = h_t  # Pass the output to the next layer
            outputs.append(h_t.unsqueeze(1))
            current_states = new_states

        outputs = torch.cat(
            outputs, dim=1
        )  # Concatenate on the time dimension
        return outputs, current_states


# Define models
# models = {
#     # "xLSTM": xLSTM(input_size, hidden_size, num_heads, batch_first=True, layers='ssm'),# msm
#     "WYD_sLSTM": WYDsLSTM(input_size, hidden_size, num_heads),
#     # "WYD_mLSTM": WYDmLSTM(input_size, hidden_size, num_heads),
#     # "LSTM": nn.LSTM(input_size, hidden_size, batch_first=True, proj_size=input_size),
#     # "GRU": GRUModel(input_size, hidden_size, num_layers=1, output_dim=1),
#     "sLSTM": sLSTM(input_size, hidden_size, num_heads, batch_first=True),
#     # "mLSTM": mLSTM(input_size, hidden_size, num_heads, batch_first=True)
# }
