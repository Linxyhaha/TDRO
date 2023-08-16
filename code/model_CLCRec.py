from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import ipdb
import pdb
import random
# from torch_geometric.utils import scatter_   # textual feature


##########################################################################

class CLCRec(torch.nn.Module):
    def __init__(self, warm_item, cold_item, num_user, num_item, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, contrastive, num_sample):
        super(CLCRec, self).__init__()
        self.num_user = num_user
        self.num_neg = num_neg
        self.contrastive = contrastive
        self.reg_weight = reg_weight
        self.temp_value = temp_value
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))))
        self.dim_feat = 0
        self.num_sample = num_sample
        
        self.emb_id = list(range(num_user)) + list(warm_item)
        self.feat_id = torch.tensor([i_id-num_user for i_id in cold_item])
        self.result = torch.zeros((num_user+num_item,dim_E)).cuda()
        
        if v_feat is not None:
            self.v_feat = F.normalize(v_feat, dim=1)
            self.dim_feat += self.v_feat.size(1)
        else:
            self.v_feat = None
        
        if a_feat is not None:
            self.a_feat = F.normalize(a_feat, dim=1)
            self.dim_feat += self.a_feat.size(1)
        else:
            self.a_feat = None

        if t_feat is not None:
            self.t_feat = F.normalize(t_feat, dim=1)
            self.dim_feat += self.t_feat.size(1)
        else:
            self.t_feat = None
        
        self.MLP = nn.Linear(dim_E, dim_E)

        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, dim_E)
        
        self.att_weight_1 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, dim_E))))
        self.att_weight_2 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, dim_E))))
        self.bias = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, 1))))
        self.att_sum_layer = nn.Linear(dim_E, dim_E)

        self.result = nn.init.kaiming_normal_(torch.rand((num_user+num_item, dim_E))).cuda()



    def feature_extractor(self):
        feature = torch.tensor([]).cuda()
        
        if self.v_feat is not None:
            feature = torch.cat((feature, self.v_feat), dim=1)
        
        if self.a_feat is not None:
            feature = torch.cat((feature, self.a_feat), dim=1)
        
        if self.t_feat is not None:
            feature = torch.cat((feature, self.t_feat), dim=1)
        
        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return feature


    def loss_contrastive(self, tensor_anchor, tensor_all, temp_value):  

        all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/(temp_value)).view(-1, 1+self.num_neg)
        all_score = all_score.view(-1, 1+self.num_neg)

        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)

        sample_loss = -torch.log(pos_score / all_score)
        contrastive_loss = sample_loss.mean()
        return contrastive_loss, sample_loss


    def forward(self, user_tensor, item_tensor):
        pos_item_tensor = item_tensor[:, 0].unsqueeze(1)
        pos_item_tensor = pos_item_tensor.repeat(1, 1+self.num_neg).view(-1, 1).squeeze()
        
        user_tensor = user_tensor.view(-1, 1).squeeze()
        item_tensor = item_tensor.view(-1, 1).squeeze()

        feature = self.feature_extractor()
        all_item_feat = feature[item_tensor-self.num_user]

        user_embedding = self.id_embedding[user_tensor]
        pos_item_embedding = self.id_embedding[pos_item_tensor]
        all_item_embedding = self.id_embedding[item_tensor]

        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0)*self.num_sample), )).cuda()
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        contrastive_loss_1, sample_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temp_value)
        contrastive_loss_2, sample_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean()+(torch.sqrt((all_item_embedding**2).sum(1))).mean())/2

        return sample_loss_1*self.contrastive, sample_loss_2 * (1-self.contrastive), reg_loss

    def loss(self, user_tensor, item_tensor):
        cf_loss, constraint_loss, reg_loss = self.forward(user_tensor, item_tensor)
        reg_loss = self.reg_weight * reg_loss
        return cf_loss+constraint_loss, reg_loss