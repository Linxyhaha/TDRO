import time
import random
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from kmeans_pytorch import kmeans

def data_load(dataset):
    dir_str = '../data/' + dataset

    train_data = np.load(dir_str+'/training_dict.npy', allow_pickle=True).item()
    val_data = np.load(dir_str+'/validation_dict.npy', allow_pickle=True).item()
    val_warm_data = np.load(dir_str+'/validation_warm_dict.npy', allow_pickle=True).item()
    val_cold_data = np.load(dir_str+'/validation_cold_dict.npy', allow_pickle=True).item()
    test_data = np.load(dir_str+'/testing_dict.npy', allow_pickle=True).item()
    test_warm_data = np.load(dir_str+'/testing_warm_dict.npy', allow_pickle=True).item()
    test_cold_data = np.load(dir_str+'/testing_cold_dict.npy', allow_pickle=True).item()

    if dataset == "amazon":
        num_user = 21607
        num_item = 93755
        num_warm_item = 75069
        pca_feat = np.load(dir_str + '/img_pca_map.npy', allow_pickle=True).item()
        v_feat = np.zeros((num_item,len(pca_feat[0]))) # pca dim = 64
        for i_id in pca_feat:
            v_feat[i_id] = np.array(pca_feat[i_id])
        v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        a_feat = None
        t_feat = None
    
    elif dataset == "micro-video":
        num_user = 21608
        num_item = 64437
        num_warm_item = 56722
        pca_feat = np.load(dir_str + '/visual_feature_64.npy', allow_pickle=True)
        v_feat = np.zeros((num_item,pca_feat.shape[1])) # pca dim = 64
        for i_id in range(num_item):
            v_feat[i_id] = pca_feat[i_id]
            i_id += 1
        v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        a_feat = None
        text_feat = np.load(dir_str + '/text_name_feature.npy', allow_pickle=True)
        t_feat = np.zeros((num_item,text_feat.shape[1]))
        for i_id in range(num_item):
            t_feat[i_id] = text_feat[i_id]
            i_id += 1
        t_feat = torch.tensor(t_feat,dtype=torch.float).cuda()

    elif dataset == "kwai":
        num_user = 7010
        num_item = 86483
        num_warm_item = 74470
        pca_feat = np.load(dir_str + '/img_pca_map.npy', allow_pickle=True)
        v_feat = torch.tensor(pca_feat,dtype=torch.float).cuda()
        a_feat = None
        t_feat = None

    else:
        raise NotImplementedError

    # add item id to org_id + num_user
    for u_id in train_data:
        for i,i_id in enumerate(train_data[u_id]):
            train_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_data[u_id]):
            val_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_warm_data[u_id]):
            val_warm_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_cold_data[u_id]):
            val_cold_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_data[u_id]):
            test_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_warm_data[u_id]):
            test_warm_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_cold_data[u_id]):
            test_cold_data[u_id][i] = i_id + num_user

    return num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat

def get_item_group_data_mask(ui_dict, portion=[3,3,3]):
    """
        This function returns the item group list, where each group contain items of different popularity.
        The group is assigned according to the popularity in test set.
    """
    item_cnt = {}
    for u_id in ui_dict:
        for i_id in ui_dict[u_id]:
            item_cnt[i_id] = item_cnt.get(i_id,0) + 1
    item_cnt = dict(sorted(item_cnt.items()))

    n_item = math.ceil(len(item_cnt)/portion[0])
    item_group = {}
    i_cnt = 0
    for i_id in item_cnt:
        item_group[i_id] = i_cnt//n_item
        i_cnt+=1
    
    group_data = [{u_id:[] for u_id in ui_dict} for _ in range(len(portion))]
    group_mask = [set() for _ in range(len(portion))]
    for u_id in ui_dict:
        for i_id in ui_dict[u_id]:
            group_data[item_group[i_id]][u_id].append(i_id)
    for i_id, g_idx in item_group.items():
        for idx in range(len(portion)):
            if idx == g_idx:
                continue
            group_mask[idx].add(i_id)

    return group_data, group_mask

def get_test_group(ui_dict, warm_item, cold_item, portion=[0.3,0.6,1]):
    """
        This function returns the user group list, where each group contain different portions of cold item interactions.
        Portion = cold interactions / all interactions (per user)
    """
    if portion[0]>1:
        group_list = [0 for u_id in range(len(ui_dict))]
        u_dict = {}
        for u_id in ui_dict:
            if len(ui_dict[u_id])==0:
                continue
            cold_cnt = 0
            for i_id in ui_dict[u_id]:
                if i_id in cold_item:
                    cold_cnt+=1
            u_coldPortion = cold_cnt/len(ui_dict[u_id]) if len(ui_dict[u_id]) else 0
            u_dict[u_id] = u_coldPortion
        u_dict = dict(sorted(u_dict.items(),key=lambda x:x[1]))

        n_user = math.ceil(len(u_dict)/portion[0])
        u_cnt = 0
        for u_id in u_dict:
            group_list[u_id] = u_cnt//n_user
            u_cnt += 1
        return group_list

    else:
        group_list = [0 for u_id in range(len(ui_dict))]
        for u_id in ui_dict:
            cold_cnt = 0
            for i_id in ui_dict[u_id]:
                if i_id in cold_item:
                    cold_cnt+=1
            u_coldPortion = cold_cnt/len(ui_dict[u_id]) if len(ui_dict[u_id]) else 0
            for g_idx, p in enumerate(portion):
                if u_coldPortion < p:
                    group_list[u_id] = g_idx
                    break
        return group_list

def get_test_group_DistSimilarity(train_dict, test_dict, warm_item, cold_item, feature, num_user, portion=[0.3,0.6,1], metric='mse'):
    """
        This function returns the user group list, where each group contain users with different stength of distribution similarity between test cold and train.
        Distribution Similarity = inner product of avg(historical interacted item features) and avg(test interacted cold item features)
    """
    # feature = F.normalize(feature, dim=1)
    group_list = [0 for u_id in range(len(train_dict))]

    u_feat_w, u_feat_c, u_distSim = {}, {}, {}
    for u_id in train_dict:
        indices = torch.LongTensor(train_dict[u_id])-num_user
        u_feat_w[u_id] = torch.mean(feature[indices],dim=0)
    for u_id in test_dict:
        if len(test_dict[u_id]):
            indices = torch.LongTensor(test_dict[u_id])-num_user
            u_feat_c[u_id] = torch.mean(feature[indices],dim=0)
    for u_id in u_feat_c:
        if metric=='cos':
            u_distSim[u_id] = (u_feat_w[u_id] @ u_feat_c[u_id] / (u_feat_w[u_id].norm()*u_feat_c[u_id].norm())).item()
        elif metric=='mse':
            u_distSim[u_id] = torch.sqrt(torch.sum((u_feat_w[u_id] - u_feat_c[u_id])**2)).item()
    u_distSim = dict(sorted(u_distSim.items(), key=lambda x:x[1], reverse=True))

    print(f"max score is {max(u_distSim.values())}")
    print(f"min score is {min(u_distSim.values())}")

    if portion[0]>1:
        n_user = math.ceil(len(u_distSim)/len(portion))
        u_cnt = 0
        for u_id in u_distSim:
            group_list[u_id] = u_cnt//n_user
            u_cnt+=1

        # average distance in each group
        for g_idx in range(portion[0]):
            print(f"average distance in Group {g_idx}: {list(u_distSim.values())[n_user*g_idx:min(n_user*g_idx+1,len(u_distSim))]}")
    else:
        for u_id in u_distSim:
            for g_idx, p in enumerate(portion[::-1]):
                if u_distSim[u_id]<p:
                    group_list[u_id] = g_idx
                    break

    return group_list

def get_test_group_popularity(ui_dict, portion=[0.3,0.6,1]):
    """
        This function returns the user group list, where each group contain different portions of interactions with popular items.
        Portion = # top 20% popular item / all interactions (per user)
    """
    if portion[0]>1:
        item_cnt = {}
        for u_id in ui_dict:
            for i_id in ui_dict[u_id]:
                item_cnt[i_id] = item_cnt.get(i_id,0) + 1
        item_cnt = dict(sorted(item_cnt.items(), key=lambda x:x[1], reverse=True))
        pop_items = list(item_cnt.keys())[:int(len(item_cnt)*0.2)] # 20% top popular items

        group_list = [0 for u_id in range(len(ui_dict))]
        u_dict = {} #{u_id:0 for u_id in ui_dict}
        for u_id in ui_dict:
            if len(ui_dict[u_id])==0:
                continue
            cold_cnt = 0
            for i_id in ui_dict[u_id]:
                if i_id in pop_items:
                    cold_cnt+=1
            u_coldPortion = cold_cnt/len(ui_dict[u_id]) if len(ui_dict[u_id]) else 0
            u_dict[u_id] = u_coldPortion
            
        n_user = math.ceil(len(u_dict)/portion[0])
        u_cnt = 0
        for u_id in u_dict:
            group_list[u_id] = u_cnt//n_user
            u_cnt += 1
        return group_list

    else:
        item_cnt = {}
        for u_id in ui_dict:
            for i_id in ui_dict[u_id]:
                item_cnt[i_id] = item_cnt.get(i_id,0) + 1
        item_cnt = dict(sorted(item_cnt.items(), key=lambda x:x[1], reverse=True))
        pop_items = list(item_cnt.keys())[:int(len(item_cnt)*0.2)] # 20% top popular items

        group_list = [0 for _ in range(len(ui_dict))]
        for u_id in ui_dict:
            cold_cnt = 0
            for i_id in ui_dict[u_id]:
                if i_id in pop_items:
                    cold_cnt+=1
            u_coldPortion = cold_cnt/len(ui_dict[u_id]) if len(ui_dict[u_id]) else 0
            for g_idx, p in enumerate(portion):
                if u_coldPortion < p:
                    group_list[u_id] = g_idx
                    break
        return group_list

class DRO_Dataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, cold_set, train_data, num_neg, n_group, n_env, split_mode='relative', pretrained_emb=None, dataset='amazon'):
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.cold_set = cold_set
        self.all_set = set(range(num_user, num_user+num_item))-self.cold_set  # all warm item
        
        self.n_group = n_group
        self.n_env = n_env

        self.dataset = dataset

        # generate group id for each interaction
        self.gen_group(pretrained_emb, n_group, train_data)

        # generate env id for each interaction
        if split_mode=='relative':
            self.gen_env_relative(train_data, n_env)
        elif split_mode=='global':
            self.gen_env_global(train_data, n_env)

    def gen_group(self,pretrained_iemb,n_group,train_data):
        rep = pretrained_iemb.cuda()
        cluster_ids = torch.zeros(self.num_item).long()
        warm_id = torch.LongTensor([wid-self.num_user for wid in self.all_set])
        if n_group>1:
            cluster_ids_, _ = kmeans(X=rep[warm_id], num_clusters=n_group, distance='euclidean', tqdm_flag=False, device=torch.device('cuda:0'))
            cluster_ids[warm_id] = cluster_ids_.detach().cpu()
        self.i_group = cluster_ids

    def ui2env(self, i_time, t_ref):
        if self.n_env==1:
            return 0
        for idxx, t_ in enumerate(t_ref):
            if i_time<=t_:
                return idxx
            if idxx == self.n_env-2:
                return idxx+1

    def gen_env_global(self, train_data, N):
        '''
            This function is used for environment splitting according to the global timestamps of the first appearance of each item.
        '''
        # generate N groups by relative time & generate a group matrix (n_user, i_item), each entry's value indicates the group id of the user-item pair
        self.train_data = []
        self.ui_env = torch.zeros((self.num_user,self.num_item)).long() 

        dir_str = f'../data/{self.dataset}'
        item_t_dict = np.load(dir_str+'/item_time_dict.npy',allow_pickle=True).item()
        id2item = np.load(dir_str+'/item_map_reverse.npy', allow_pickle=True).item()

        # equal item in each group
        t_list = []
        for i_id in self.all_set:
            t_list.append(item_t_dict[id2item[i_id-self.num_user]])
        t_sorted = sorted(t_list)

        # group split idx
        length = len(t_sorted)
        n_t = length//N
        t_ref = []
        for idx in range(N-1):
            t_ref.append(t_sorted[(idx+1)*n_t])
        
        env_sample_num = {e_id:0 for e_id in range(self.n_env)}
        group_sample_num = {g_id:0 for g_id in range(self.n_group)}

        # assign groups to interaction pairs
        for u_id, i_ids in train_data.items():
            for i_id in i_ids:
                t_i = item_t_dict[id2item[i_id-self.num_user]]
                e_id = self.ui2env(t_i, t_ref)
                env_sample_num[e_id] += 1
                g_id = self.i_group[i_id-self.num_user].item()
                group_sample_num[g_id] += 1
                self.train_data.append([u_id,i_id,e_id,g_id])
                self.ui_env[u_id][i_id-self.num_user] = e_id
        for e in env_sample_num:
            print(f"interaction num of Environment {e}: {env_sample_num[e]}")
        for g in group_sample_num:
            print(f"interaction num of Group {g}: {group_sample_num[g]}")

    def gen_env_relative(self, train_data, N):
        '''
            This function is used for environment splitting according to the relative timestamps of interactions for each user.
        '''
        # generate N groups by relative time & generate a group matrix (n_user, i_item), each entry's value indicates the group id of the user-item pair
        self.train_data = []
        self.ui_env = torch.zeros((self.num_user,self.num_item)).long()
        env_sample_num = {e_id:0 for e_id in range(self.n_env)}
        group_sample_num = {g_id:0 for g_id in range(self.n_group)}
        for u_id, i_ids in train_data.items():
            g_len = len(i_ids)//N
            # if it is not enough for each group having at least one interaction
            if g_len==0: 
                for i_idx, i_id in enumerate(i_ids[::-1]):
                    e_id = N-i_idx-1
                    env_sample_num[e_id] += 1
                    g_id = self.i_group[i_id-self.num_user].item()
                    group_sample_num[g_id] += 1
                    self.train_data.append([u_id,i_id,e_id,g_id])
                    self.ui_env[u_id][i_id-self.num_user] = e_id
            # if each group would have at least one interaction
            else: 
                for i_idx, i_id in enumerate(i_ids[::-1]):
                    e_id = N-i_idx//g_len-1 # from back to the front, if more, env 0 has more interactions that belongs to env<0
                    e_id = 0 if e_id<0 else e_id
                    env_sample_num[e_id] += 1
                    g_id = self.i_group[i_id-self.num_user].item()
                    group_sample_num[g_id] += 1
                    self.train_data.append([u_id,i_id,e_id,g_id])
                    self.ui_env[u_id][i_id-self.num_user] = e_id
        for e in env_sample_num:
            print(f"interaction num of Environment {e}: {env_sample_num[e]}")
        for g in group_sample_num:
            print(f"interaction num of Group {g}: {group_sample_num[g]}")

            
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, pos_item, e_pos, g_pos = self.train_data[index]
        neg_item = random.sample(self.all_set-set(self.user_item_dict[user]), self.num_neg)

        user_tensor = torch.LongTensor([user]*(self.num_neg+1))
        item_tensor = torch.LongTensor([pos_item] + neg_item)

        group_tensor = torch.LongTensor([g_pos])
        env_tensor = torch.LongTensor([e_pos])

        return user_tensor, item_tensor, group_tensor, env_tensor
