import argparse
import os
import time
import numpy as np
import torch
import random
from Dataset import data_load, DRO_Dataset
from model_CLCRec import CLCRec
from torch.utils.data import DataLoader
from Train import train_TDRO
from Full_rank import full_ranking
from Metric import print_results

###############################248###########################################

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='kwai_small', help='Dataset path')
    parser.add_argument('--model_name', default='SSL', help='Model Name.')
    parser.add_argument('--log_name', default='', help='log name.')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument("--topK", default='[10, 20, 50, 100]', help="the recommended item num")
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--dim_E', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='Weight decay.')

    # CLCRec
    parser.add_argument('--num_neg', type=int, default=256, help='Negative size.')
    parser.add_argument('--contrastive', type=float, default=0.1, help='Weight loss one.')
    parser.add_argument('--num_sample', type=float, default=0.5, help='probability of robust training.')
    parser.add_argument('--temp_value', type=float, default=0.1, help='Contrastive temp_value.')
    
    parser.add_argument('--pretrained_emb', type=str, default='./pretrained_emb/', help='path of pretrained embedding of items')

    # Group-DRO
    parser.add_argument('--num_group', type=int, default=1, help='group number for group DRO')
    parser.add_argument('--mu', type=float, default=0.5, help='streaming learning rate for group DRO')
    parser.add_argument('--eta', type=float, default=0.01, help='step size for group DRO')

    # TDRO
    parser.add_argument('--num_period', type=int, default=1, help='time period number for TDRO')
    parser.add_argument('--split_mode', type=str, default='global', help='split the group by global time or relative interactions per user', choices=['relative','global'])
    parser.add_argument('--lam', type=float, default=0.5, help='coefficient for time-aware shifting trend')
    parser.add_argument('--p', type=float, default=0.2, help='strength of gradient (see more in common good)')

    # group evaluation
    parser.add_argument('--group_test', action='store_true', help='whether or not do evaluation of user/item groups')
    parser.add_argument('--portion_list', type=str, help='portion list of different groups')

    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--save_path', default='./models/', help='model save path')
    parser.add_argument('--inference',action='store_true', help='only inference stage')
    parser.add_argument('--ckpt', type=str, help='pretrained model path')
    
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

    num_group = args.num_group
    num_period = args.num_period
    mu = args.mu
    eta = args.eta
    lam = args.lam
    p = args.p
    
    temp_value = args.temp_value
    step = args.step
    portion_list = eval(args.portion_list) if args.inference else None

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

    train_dataset = DRO_Dataset(num_user, num_item, user_item_all_dict, cold_item, train_data, num_neg, num_group, num_period, split_mode, pretrained_emb)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    
    print('Data has been loaded.')

    ##########################################################################################################################################
    model = CLCRec(warm_item, cold_item, num_user, num_item, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, contrastive, num_sample).cuda()
    ##########################################################################################################################################
    if args.inference:
        with torch.no_grad():
            model = torch.load('models/' + args.ckpt)
            test_result = full_ranking(0, model, test_data, tv_dict, None, False, step, topK)
            test_result_warm = full_ranking(0, model, test_warm_data, tv_dict, cold_item, False, step, topK)
            test_result_cold = full_ranking(0, model, test_cold_data, tv_dict, warm_item, False, step, topK)
            print('---'*18)
            print('All')
            print_results(None,None,test_result)
            print('Warm')
            print_results(None,None,test_result_warm)
            print('Cold')
            print_results(None,None,test_result_cold)
        os._exist(1)
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])#, 'weight_decay': reg_weight}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0 
    best_epoch = 0
    max_val_result = max_val_result_warm = max_val_result_cold = max_test_result = max_test_result_warm = max_test_result_cold = None
    w_list = torch.ones(num_group).cuda() 
    loss_list = torch.zeros(num_group).cuda()
    for epoch in range(num_epoch):
        epoch_start_time = time.time()

        loss = train_TDRO(train_dataloader, model, optimizer, num_group, num_period, loss_list, w_list, mu, eta, lam, p)
        
        elapsed_time = time.time() - epoch_start_time
        print("Train: The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        if torch.isnan(loss):
            print("Loss is Nan. Quit.")
            break
        torch.cuda.empty_cache()
        if (epoch+1)%1==0:    
            test_result = None

            with torch.no_grad():
                val_result = full_ranking(model, val_data, train_dict, None, False, step, topK)
                test_result = full_ranking(model, test_data, tv_dict, None, False, step, topK)
                test_result_warm = full_ranking(model, test_warm_data, tv_dict, cold_item, False, step, topK)
                test_result_cold = full_ranking(model, test_cold_data, tv_dict, warm_item, False, step, topK)
                
            print('---'*18)
            print("Runing Epoch {:03d} ".format(epoch) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-epoch_start_time)))
            print_results(None,val_result,test_result)
            print_results(None,None,test_result_warm)
            print_results(None,None,test_result_cold)

            print('---'*18)

            if val_result[1][0] > max_recall:
                best_epoch=epoch
                pre_id_embedding = model.id_embedding
                max_recall = val_result[1][0]
                max_val_result = val_result

                max_test_result = test_result
                max_test_result_warm = test_result_warm
                max_test_result_cold = test_result_cold
                num_decreases = 0
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path) 
                torch.save(model, '{}{}_{}_{}lr_{}reg_{}ng_{}con_{}rou_{}temp_{}dimE_{}G_{}E_{}mu_{}eta_{}lam_{}p_{}_{}.pth'.format(args.save_path, \
                                    args.model_name, args.data_path, args.l_r, args.reg_weight, args.num_neg, args.contrastive, \
                                        args.num_sample, args.temp_value, args.dim_E, args.num_group, args.num_period, \
                                            args.mu, args.eta, args.lam, args.p, args.split_mode, args.log_name))
            else:
                if num_decreases > 5:
                    print('-'*18)
                    print('Exiting from training early')
                    break
                else:
                    num_decreases += 1
    model = torch.load('{}{}_{}_{}lr_{}reg_{}ng_{}con_{}rou_{}temp_{}dimE_{}G_{}E_{}mu_{}eta_{}lam_{}p_{}_{}.pth'.format(args.save_path, \
                                    args.model_name, args.data_path, args.l_r, args.reg_weight, args.num_neg, args.contrastive, \
                                        args.num_sample, args.temp_value, args.dim_E, args.num_group, args.num_period, \
                                            args.mu, args.eta, args.lam, args.p, args.split_mode, args.log_name))
    model.eval()
    with torch.no_grad():
        test_result = full_ranking(model, test_data, tv_dict, None, False, step, topK)
        test_result_warm = full_ranking(model, test_warm_data, tv_dict, cold_item, False, step, topK)
        test_result_cold = full_ranking(model, test_cold_data, tv_dict, warm_item, False, step, topK)

    print('==='*18)
    print(f"End. Best Epoch is {best_epoch}")
    print('---'*18)
    print('Validation')
    print_results(None,max_val_result,max_test_result)
    print('Test')
    print('All')
    print_results(None, None, test_result)
    print('Warm')
    print_results(None,None,test_result_warm)
    print('Cold')
    print_results(None,None,test_result_cold)
    print('---'*18)