import warnings
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
warnings.filterwarnings("ignore")


class TicTacToeModel(pl.LightningModule):
    def __init__(self, board_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = 0.001

        # basic network, should be enough to approximate Q-table. Have tried much more complex models, same result
        self.input_layer = nn.Linear(board_size**2, board_size**2)
        self.hidden_layer = nn.Linear(board_size**2, board_size**2)
        self.output_layer = nn.Linear(board_size**2, board_size**2)

    def forward(self, t):
        # rely to add non-linearity
        t = F.relu(self.input_layer(t))
        t = F.relu(self.hidden_layer(t))
        t = self.output_layer(t)
        return t

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # standard loss for regression
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_hat, y)

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

    def train_dataloader(self):
        static_q_table = StaticQTable()
        train_dataloader = DataLoader(dataset=static_q_table, batch_size=4, shuffle=True, num_workers=1)
        return train_dataloader


class StaticQTable(Dataset):
    def __init__(self):
        super().__init__()
        self.q = np.load("10000_games.npy", allow_pickle=True)
        t = 1

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, index):
        state, q_values = torch.Tensor(self.q[index][0]), torch.Tensor(self.q[index][1])
        if not int(torch.max(torch.abs(q_values))) == 0:
            q_values = torch.div(q_values, torch.max(torch.abs(q_values)))
        return [state, q_values]


if __name__ == "__main__":
    # example state, -1 = O, 1 = X
    state = torch.Tensor([0., 0., 0., 0., 0., 0., -1., 0., 1.])

    net = TicTacToeModel()
    trainer = pl.Trainer(min_epochs=1, max_epochs=3, gpus=1)
    trainer.fit(net)
    torch.save(net.state_dict(), "static_model.pth")
    # net.load_state_dict(torch.load("static_model.pth"))

    net.eval()
    output = net(state)
    print(output)
    # Q-value expected output from network
    print("tensor([ 1.0000,  0.3066,  0.6045, -0.0081,  0.7336,  0.3203,  0.0000, -0.2233, 0.0000])")
    t = 1
