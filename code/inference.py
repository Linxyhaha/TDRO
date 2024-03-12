import argparse
import os
import time
import numpy as np
import torch
import random
from Dataset import data_load, DRO_Dataset, get_test_group_popularity, get_item_group_data_mask
from model_CLCRec import CLCRec
from torch.utils.data import DataLoader
from Full_rank import full_ranking
from Metric import print_results

###############################248###########################################

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--data_path', default='kwai_small', help='Dataset path')
    parser.add_argument('--model_name', default='SSL', help='Model Name.')
    parser.add_argument('--log_name', default='', help='log name.')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument("--topK", default='[10, 20, 50, 100]', help="the recommended item num")
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--num_neg', type=int, default=512, help='Negative size.')
    parser.add_argument('--contrastive', type=float, default=1, help='Weight loss one.')
    parser.add_argument('--temp_value', type=float, default=1, help='Contrastive temp_value.')
    parser.add_argument('--reg_weight', type=float, default=1e-1, help='Weight decay.')
    parser.add_argument('--num_sample', type=float, default=0.5, help='probability of robust training.')

    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument("--no_cuda", action="store_true", help="disable the cuda")
    parser.add_argument('--save_path', default='./models/', help='model save path')
    parser.add_argument('--inference',action='store_true', help='only inference stage')
    parser.add_argument('--ckpt', type=str, help='pretrained model path')

    parser.add_argument('--pretrained_emb', type=str, default='./pretrained_emb/', help='path of pretrained embedding of items')

    # Group-DRO
    parser.add_argument('--num_group', type=int, default=1, help='group number for group DRO')
    parser.add_argument('--mu', type=float, default=0.5, help='streaming learning rate for group DRO')
    parser.add_argument('--eta', type=float, default=0.01, help='step size for group DRO')

    # TDRO
    parser.add_argument('--num_period', type=int, default=1, help='environment number for group DRO')
    parser.add_argument('--split_mode', type=str, default='relative', help='split the group by global time or relative interactions per user', choices=['relative','global'])
    parser.add_argument('--lam', type=float, default=0.5, help='coefficient for time-aware shifting trend')
    parser.add_argument('--p', type=float, default=0.5, help='strength of gradient (see more in common good)')
    
    # group evaluation
    parser.add_argument('--group_test', action='store_true', help='whether or not do evaluation of user/item groups')
    parser.add_argument('--group_type', type=str, default='user', help='user group or item group evaluation')
    parser.add_argument('--group_metric', type=str, default='mse', help='the metric for group splitting')
    parser.add_argument('--portion_list', type=str, help='portion list of different groups')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init()
    print(args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path


    learning_rate = args.l_r
    contrastive = args.contrastive
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    split_mode = args.split_mode
    num_neg = args.num_neg
    num_sample = args.num_sample
    topK = eval(args.topK)

    temp_value = args.temp_value
    step = args.step

    dim_E = args.dim_E
    ##########################################################################################################################################
    print('Data loading ...')

    num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat = data_load(data_path)
    dir_str = f'../data/{args.data_path}'

    user_item_all_dict = {}
    train_dict = {}
    tv_dict = {}
    for u_id in train_data:
        user_item_all_dict[u_id] = train_data[u_id] + val_data[u_id] + test_data[u_id]
        train_dict[u_id] = train_data[u_id]
        tv_dict[u_id] = train_data[u_id] + val_data[u_id]
    warm_item = torch.tensor(list(np.load(dir_str + '/warm_item.npy',allow_pickle=True).item()))
    cold_item = torch.tensor(list(np.load(dir_str + '/cold_item.npy',allow_pickle=True).item()))
    warm_item = set([i_id.item() + num_user for i_id in warm_item])    # item id = item_id_org + user num
    cold_item = set([i_id.item() + num_user for i_id in cold_item])

    # pretrained item embedding
    pretrained_emb = np.load(args.pretrained_emb+data_path+'/all_item_feature.npy',allow_pickle=True)
    pretrained_emb = torch.FloatTensor(pretrained_emb).cuda()

    train_dataset = DRO_Dataset(num_user, num_item, user_item_all_dict, cold_item, train_data, num_neg, args.num_group, args.num_period, split_mode, pretrained_emb)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    
    print('Data has been loaded.')
    ##########################################################################################################################################
    # inference stage
    if args.inference:
        with torch.no_grad():
            model = torch.load('models/' + args.ckpt)
            test_result = full_ranking(model, test_data, tv_dict, None, False, step, topK)
            test_result_warm = full_ranking(model, test_warm_data, tv_dict, cold_item, False, step, topK)
            test_result_cold = full_ranking(model, test_cold_data, tv_dict, warm_item, False, step, topK)
            print('---'*18)
            print('All')
            print_results(None,None,test_result)
            print('Warm')
            print_results(None,None,test_result_warm)
            print('Cold')
            print_results(None,None,test_result_cold)
        os._exit(1)