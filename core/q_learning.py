import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import logging
import sys

sys.path.append("../")
from core.other_functions import *

mylogger = logging.getLogger("testSA2-fqe")


class RLDataset(Dataset):
    def __init__(self, S, A, R):
        self.s_dim = S.shape[2]
        self.N, self.T = A.shape
        self.S = S[:, :self.T].reshape([self.N * self.T, self.s_dim])
        self.Sp = S[:, 1:(self.T + 1)].reshape([self.N * self.T, self.s_dim])
        self.A = A.reshape([self.N * self.T, 1])
        self.R = R.reshape([self.N * self.T, 1])

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        S = torch.tensor(self.S[idx], dtype=torch.float32)
        Sp = torch.tensor(self.Sp[idx], dtype=torch.float32)
        A = torch.tensor(self.A[idx], dtype=torch.int64)
        R = torch.tensor(self.R[idx], dtype=torch.float32)
        return S, A, R, Sp


# implementation of offline DQN
class QNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None):
        super(QNet, self).__init__()
        if hidden_dims == None:
            hidden_dims = [32, 64, 32]
        nn_dims = [in_dim] + hidden_dims + [out_dim]
        modules = []
        for i in range(len(nn_dims) - 1):
            if i == len(nn_dims) - 2:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1])))
            else:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1]),
                                  nn.Sigmoid()))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class dqn_model(nn.Module):
    def __init__(self, model_parameters, state_normalizer) -> None:
        super().__init__()
        self.Q = QNet(in_dim=model_parameters["in_dim"],
                      out_dim=model_parameters["out_dim"],
                      hidden_dims=model_parameters["hidden_dims"])
        self.target_Q = QNet(in_dim=model_parameters["in_dim"],
                             out_dim=model_parameters["out_dim"],
                             hidden_dims=model_parameters["hidden_dims"])
        self.update_target_net()
        self.gamma = model_parameters["gamma"]
        self.state_normalizer = state_normalizer

    def loss_function(self, s, a, r, sp):
        state_action_values = self.Q(s).gather(1, a).flatten()
        _, a_selected = self.Q(sp).detach().max(1)
        next_state_values = self.target_Q(sp).gather(
            1, a_selected.reshape([-1, 1])).detach().flatten()
        target_values = r.flatten() + self.gamma * next_state_values
        td_error = nn.MSELoss()(target_values, state_action_values)
        return td_error
    
    def update_target_net(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def act(self, s):
        s_n = self.state_normalizer.normalize(s)
        if not torch.is_tensor(s_n):
            s_n = torch.tensor(s_n, dtype=torch.float32)
        else:
            s_n = s_n.to(torch.float32)
        with torch.no_grad():
            self.Q.eval()
            action = torch.argmax(self.Q(s_n), dim=1)
        return action.cpu().detach().numpy()

    def __call__(self, s):
        return self.act(s)


def offline_dqn_learning(data,
                         model_parameters,
                         train_parameters,
                         ts_writer=None):
    # torch.set_num_threads(1)
    ## extract parameters
    S, A, R = data["S"], data["A"], data["R"]
    ## normalizatin
    state_normalizer = array_normalizer(data=S.reshape([-1, S.shape[-1]]),
                                        axis=(0))
    reward_noemalizer = array_normalizer(data=R, axis=(0, 1))
    S_n = state_normalizer.normalize(S)
    R_n = reward_noemalizer.normalize(R)

    ## set parameter
    in_dim = model_parameters["in_dim"]
    out_dim = model_parameters["out_dim"]
    hidden_dims = model_parameters["hidden_dims"]
    batch_size, epochs, target_update_freq, learning_rate,grad_norm = train_parameters[
        "batch_size"], train_parameters["epochs"], train_parameters[
            "target_update_freq"], train_parameters["learning_rate"],train_parameters["grad_norm"]
    ## make dataloader
    dataloader_train = torch.utils.data.DataLoader(RLDataset(S_n, A, R_n),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=False)

    ## pre training
    device = torch.device("cpu")
    model = dqn_model(model_parameters=model_parameters,
                      state_normalizer=state_normalizer).to(device)
    optimizer = optim.Adam(model.Q.parameters(), lr=learning_rate)  # 0.001
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=3000000,
                                          gamma=0.5)

    model.Q.train()
    model.target_Q.eval()
    iters = 1
    num_updates = 0
    for ep in range(epochs):
        for (batch_idx, batch) in enumerate(dataloader_train):
            s, a, r, sp = batch[0],batch[1],batch[2],batch[3]
            optimizer.zero_grad()
            loss = model.loss_function(s, a, r, sp)  # problem is here
            loss.backward()
            nn.utils.clip_grad_norm_(model.Q.parameters(), grad_norm)
            optimizer.step()
            scheduler.step()
            if iters % target_update_freq == 0:
                model.update_target_net()
                num_updates +=1
                if ts_writer:
                    ts_writer.add_scalar('DQN update_{}'.format(S.shape[1]), num_updates, iters)
            if iters % 100 == 0:
                if ts_writer:
                    ts_writer.add_scalar('DQN training loss_{}'.format(S.shape[1]), loss.item(), iters)
                    grad_norm_now = nn.utils.clip_grad_norm_(model.Q.parameters(), 10000000)
                    ts_writer.add_scalar('DQN grad norm_{}'.format(S.shape[1]),grad_norm_now,iters)
            iters += 1
    return (model.to("cpu"))


def calculate_expected_discounted_reward_MC(reward, gamma):
    N, T = reward.shape
    s = np.zeros([N])
    for t in range(T - 1, -1, -1):
        s = gamma * s + reward[:, t]
    return np.mean(s)


if __name__ == "__main__":
    pass