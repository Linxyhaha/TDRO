import torch
import math 

def rank(num_user, user_item_inter, mask_items, result, is_training, step, topk):
    user_tensor = result[:num_user]
    item_tensor = result[num_user:]
    start_index = 0
    end_index = num_user if step==None else step
    all_index_of_rank_list = torch.LongTensor([])
    while end_index <= num_user and start_index < end_index:
        temp_user_tensor = user_tensor[start_index:end_index]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        if is_training is False: # mask training interactions
            for row, col in user_item_inter.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-num_user
                    score_matrix[row][col] = -1e15
            if mask_items is not None:
                score_matrix[:, mask_items-num_user] = -1e15

        _, index_of_rank_list = torch.topk(score_matrix, topk)
        all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+num_user), dim=0)
        start_index = end_index
        if end_index+step < num_user:
            end_index += step
        else:
            end_index = num_user
    return all_index_of_rank_list
    
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        user_length = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                user_length += 1
        
        precision.append(round(sumForPrecision / user_length, 4))
        recall.append(round(sumForRecall / user_length, 4))
        NDCG.append(round(sumForNdcg / user_length, 4))
        MRR.append(round(sumForMRR / user_length, 4))
        
    return precision, recall, NDCG, MRR

def group_computeTopNAccuracy(GroundTruth, predictedIndices, groupIndices, topN):
    """
        groupIndices is a list that contain the group idx in accordance with the GroundTruth list and predictedIndices list.
    """
    num_group = max(groupIndices)+1

    precision = [[] for _ in range(num_group)] 
    recall = [[] for _ in range(num_group)] 
    NDCG = [[] for _ in range(num_group)] 
    MRR = [[] for _ in range(num_group)] 

    for index in range(len(topN)):
        sumForPrecision = [0] * num_group
        sumForRecall = [0] * num_group
        sumForNdcg = [0] * num_group
        sumForMRR = [0] * num_group
        user_length = [0] * num_group
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                
                sumForPrecision[groupIndices[i]] += userHit / topN[index]
                sumForRecall[groupIndices[i]] += userHit / len(GroundTruth[i])               
                sumForNdcg[groupIndices[i]] += ndcg
                sumForMRR[groupIndices[i]] += userMRR
                user_length[groupIndices[i]] += 1

        for g_idx in range(num_group):
            precision[g_idx].append(round(sumForPrecision[g_idx] / user_length[g_idx], 4))
            recall[g_idx].append(round(sumForRecall[g_idx] / user_length[g_idx], 4))
            NDCG[g_idx].append(round(sumForNdcg[g_idx] / user_length[g_idx], 4))
            MRR[g_idx].append(round(sumForMRR[g_idx] / user_length[g_idx], 4))

    return [(precision[g_idx], recall[g_idx], NDCG[g_idx], MRR[g_idx]) for g_idx in range(num_group)]
    
def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))