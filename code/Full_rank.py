import torch
from torch.autograd import no_grad

from Metric import rank, computeTopNAccuracy, group_computeTopNAccuracy

def full_ranking(model, data, user_item_inter, mask_items, is_training, step, topk, group=False): 
    model.eval()
    if mask_items is not None:
        mask_items = torch.LongTensor(list(mask_items))
    with no_grad():               
        all_index_of_rank_list = rank(model.num_user, user_item_inter, mask_items, model.result, is_training, step, topk[-1])
        gt_list = [None for _ in range(model.num_user)]
        for u_id in data:
            gt_list[u_id] = data[u_id]
        if group:
            results = group_computeTopNAccuracy(gt_list, all_index_of_rank_list, group, topk)
        else:
            results = computeTopNAccuracy(gt_list, all_index_of_rank_list, topk)
        return results