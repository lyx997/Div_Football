import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
import numpy as np

def shuffle(data, batch_size):
    batch = {}
    shuffle_index = torch.randint(0, data["player_div"].shape[1], (batch_size,))
    for key, value in data.items():
        if key != "hidden_div":
            batch[key] = value[:, shuffle_index, :].cuda()
        else:
            hidden1 = value[0][:, shuffle_index, :].cuda()
            hidden2 = value[1][:, shuffle_index, :].cuda()
            batch[key] = (hidden1, hidden2)
    return batch


class Algo():
    def __init__(self, arg_dict, device=None):
        self.K_epoch = arg_dict["k_epoch"]
        self.batch_num = arg_dict["batch_num"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]
        self.batch_size = arg_dict["div_batch_size"]

    def train(self, div_model, data):
        div_loss_lst = []
        div_entropy_lst = []
        for i in range(self.batch_num):
            s = shuffle(data, self.batch_size)
            for j in range(self.K_epoch):

                    prob_z, _ = div_model(s)

                    label_z = s["skill"]
                    log_prob_z = - torch.log(prob_z+1e-8)
                    div_entropy = torch.diagonal(torch.mm(prob_z, log_prob_z.permute(1,0)), dim1=0, dim2=1)


                    #div_loss = F.smooth_l1_loss(prob_z, label_z)
                    div_loss = F.smooth_l1_loss(prob_z.unsqueeze(0), label_z)
                    #div_entropy_loss = -1*self.entropy_coef*div_entropy
                    #loss = att_loss + att_entropy_loss.mean() 
                    #loss = loss.mean()

                    div_model.optimizer.zero_grad()
                    div_loss.backward()
                    nn.utils.clip_grad_norm_(div_model.parameters(), self.grad_clip)
                    div_model.optimizer.step()

                    div_loss_lst.append(div_loss.item())
                    div_entropy_lst.append(div_entropy.mean().item())
               
        return np.mean(div_loss_lst), np.mean(div_entropy_lst)
