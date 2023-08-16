import torch
import torch.nn as nn

import math
import scipy.optimize as sopt
import torch.nn.functional as F
from torch.autograd import grad

def train_ERM(dataloader, model, optimizer):    
    model.train()
    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()
        loss = model.loss(user_tensor.cuda(), item_tensor.cuda())
        loss.backward()
        optimizer.step()
    return loss

def train_TDRO(dataloader, model, optimizer, n_group, n_period, loss_list, w_list, mu, eta, lamda, beta_p):

    model.train()

    # period importance
    m = nn.Softmax(dim=1) 
    beta_e = m(torch.tensor([math.exp(beta_p * e) for e in range(n_period)]).unsqueeze(0).unsqueeze(-1).cuda())


    for user_tensor, item_tensor, group_tensor, period_tensor in dataloader:
        optimizer.zero_grad()

        sample_loss, reg_loss = model.loss(user_tensor.cuda(), item_tensor.cuda())

        # calculate each group-period loss and group-period gradient
        loss_ge = torch.zeros((n_group,n_period)).cuda()
        grad_ge = torch.zeros((n_group,n_period,model.encoder_layer2.weight.reshape(-1).size(0))).cuda() #(linear layer input * output)
        for name, param in model.named_parameters():
            if name == 'encoder_layer2.weight':
                for g_idx in range(n_group):
                    for e_idx in range(n_period):
                        indices = ((group_tensor.squeeze(1))==g_idx)&(period_tensor.squeeze(1)==e_idx)
                        de = torch.sum(indices)
                        loss_single = torch.sum(sample_loss*(indices).cuda())
                        grad_single = grad(loss_single, param, retain_graph=True)[-1].reshape(-1) # linear layer input*output
                        grad_single = grad_single/(grad_single.norm()+1e-16) * torch.pow(loss_single/(de+1e-16), 1) 
                        loss_ge[g_idx,e_idx] = loss_single
                        grad_ge[g_idx,e_idx] = grad_single
                        
        # worst-case factor
        de = torch.tensor([torch.sum(group_tensor==g_idx) for g_idx in range(n_group)]).cuda()
        loss_ = torch.sum(loss_ge,dim=1)
        loss_ = loss_/(de+1e-16)
        
        # shifting factor
        trend_ = torch.zeros(n_group).cuda()
        for g_idx in range(n_group):
            g_j = torch.mean(grad_ge[g_idx],dim=0) # sum up the period gradient for group 
            sum_gie = torch.mean(grad_ge * beta_e, dim=[0,1])
            trend_[g_idx] = g_j@sum_gie

        loss_ = loss_ * (1-lamda) + trend_ * lamda

        # loss consistency enhancement
        loss_[loss_==0] = loss_list[loss_==0]
        loss_list = (1 - mu) * loss_list + mu * loss_ 

        # group importance smoothing
        update_factor = eta * loss_list
        w_list = w_list * torch.exp(update_factor)
        w_list = w_list/torch.sum(w_list)
        loss_weightsum = torch.sum(w_list * loss_list)

        # add regularization loss
        loss_weightsum = loss_weightsum + reg_loss

        # back propagation and update parameters
        loss_weightsum.backward()
        optimizer.step()

        loss_list.detach_()
        w_list.detach_()

    with torch.no_grad():
        model.result[model.emb_id] = model.id_embedding[model.emb_id].data
        model.result[model.feat_id + model.num_user] = model.feature_extractor()[model.feat_id].data

    return loss_weightsum