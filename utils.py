import torch
import random
import numpy as np
from scipy.interpolate import interp1d
 
def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True 

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

def select_seed(bkg_score, point_anno):
    point_anno_agnostic = point_anno.max(dim=2)[0]
    bkg_seed = torch.zeros_like(point_anno_agnostic)
    act_seed = point_anno.clone().detach()

    act_thresh = 0.1  
    bkg_thresh = 0.95

    for b in range(point_anno.shape[0]):
        act_idx = torch.nonzero(point_anno_agnostic[b]).squeeze(1)
        if len(act_idx) == 0:
            continue
        """ most left """
        if act_idx[0] > 0:
            bkg_score_tmp = bkg_score[b, :act_idx[0]]
            idx_tmp = bkg_seed[b, :act_idx[0]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[:start_index] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[:max_index + 1] = 1
            """ pseudo action point selection """
            for j in range(act_idx[0] - 1, -1, -1):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[0]]
                else:
                    break

        """ most right """
        if act_idx[-1] < (point_anno.shape[1] - 1):
            bkg_score_tmp = bkg_score[b, act_idx[-1] + 1:]
            idx_tmp = bkg_seed[b, act_idx[-1] + 1:]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                idx_tmp[start_index:] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index:] = 1
            """ pseudo action point selection """
            for j in range(act_idx[-1] + 1, point_anno.shape[1]):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[-1]]
                else:
                    break

        """ between two instances """
        for i in range(len(act_idx) - 1):
            if act_idx[i + 1] - act_idx[i] <= 1:
                continue

            bkg_score_tmp = bkg_score[b, act_idx[i] + 1:act_idx[i + 1]]
            idx_tmp = bkg_seed[b, act_idx[i] + 1:act_idx[i + 1]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 2:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                end_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[start_index + 1:end_index] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index] = 1
            """ pseudo action point selection """
            for j in range(act_idx[i] + 1, act_idx[i + 1]):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[i]]
                else:
                    break
            for j in range(act_idx[i + 1] - 1, act_idx[i], -1):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[i + 1]]
                else:
                    break

    return act_seed, bkg_seed

def extract_region_feat(seq, embeded_feature): 
    '''
    Extract region features.
    Input: seq:[0,1,1,0,...,0,1,1,0] embeded_feature: [T,F]
    Output: feature list:[[T1,F],[T2,F],...]
    '''
    seq_diff = seq[1:] - seq[:-1]
    range_idx = torch.nonzero(seq_diff).squeeze(1)
    range_idx = range_idx.cpu().data.numpy().tolist()
    if len(range_idx) == 0:
        return
    if seq_diff[range_idx[0]] != 1:
        range_idx = [-1] + range_idx
    if seq_diff[range_idx[-1]] != -1:
        range_idx = range_idx + [seq_diff.shape[0] - 1]

    feature_lsts = []
    idx = []
    for i in range(len(range_idx) // 2):
        if range_idx[2 * i + 1] - range_idx[2 * i] < 1:
            continue
        feature_lsts.append(embeded_feature[range_idx[2 * i] + 1:range_idx[2 * i + 1] + 1].clone())
        idx.append([range_idx[2 * i] + 1, range_idx[2 * i + 1] + 1])
    return feature_lsts

def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale

def get_proposal_oic(args, tList, wtcam, vid_score, c_pred, v_len, num_segments, v_duration):
    t_factor = float(16 * v_len) / ( args.scale * num_segments * args.frames_per_sec )
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - args._lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + args._lambda * len_proposal))
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + \
                                    list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + args.gamma * vid_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([t_start, t_end, c_pred[i], c_score])
            temp.append(c_temp)
    return temp

def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

def post_process(args, vid_name, proposal_dict, test_loader):
    final_proposals = []
    for class_id in proposal_dict.keys():
        temp_proposal = soft_nms(proposal_dict[class_id], sigma=0.3)
        final_proposals += temp_proposal
    if args.dataset == "THUMOS14":
        ambilist = test_loader.dataset.ambilist
        final_proposals = np.array(final_proposals)
        final_proposals = filter_segments(final_proposals, vid_name, ambilist)
    final_proposals = result2json(args, final_proposals)

    return final_proposals

def soft_nms(dets, iou_thr=0.7, method='gaussian', sigma=0.3):
    """
    Apply Soft NMS to a set of detection results.
    """
    # expand dets with areas, and the second dimension is
    # x1, x2, label, score, area
    dets = np.array(dets)
    areas = dets[:, 1] - dets[:, 0] + 1
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 3], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1].tolist())

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        xx2 = np.minimum(dets[0, 1], dets[1:, 1])
        inter = np.maximum(xx2 - xx1 + 1, 0.0)
        iou = inter / (dets[0, -1] + dets[1:, -1] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 3] *= weight
        dets = dets[1:, :]

    return retained_box

def filter_segments(segment_predict, vn, ambilist):
    """
    Filter out segments overlapping with ambiguous_test segments.
    """
    num_segment = len(segment_predict)
    ind = np.zeros(num_segment)
    for i in range(num_segment):
        for a in ambilist:
            if a[0] == vn:
                gt = range(int(round(float(a[2]) )), int(round(float(a[3]) )))
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(len(set(gt).union(set(pd))))
                if IoU > 0:
                    ind[i] = 1
    s = [segment_predict[i, :] for i in range(num_segment) if ind[i] == 0]
    return np.array(s)

def result2json(args, result):
    result_file = []
    for i in range(len(result)):
        line = {
            'label': args.class_name_lst[int(result[i][2])],
            'score': result[i][3],
            'segment': [result[i][0], result[i][1]]
        }
        result_file.append(line)
    return result_file

def get_prediction(proposals, data_dict, dataset):
    t_factor = dataset.t_factor
    proposal_dict = {}
    prop_iou = data_dict['iou_pred_orig'][0].cpu().numpy()
    for i in range(proposals.shape[0]):
        c = int(proposals[i,3])
        if c not in proposal_dict.keys():
            proposal_dict[c] = []
        c_score = prop_iou[i, 0] + proposals[i, 2]
        proposal_dict[c].append([proposals[i, 0] / t_factor, proposals[i, 1] / t_factor, c, c_score])

    prop_iou = data_dict['iou_pred_refined'][0].cpu().numpy()
    proposals = data_dict['prop_refined'][0].cpu().numpy()
    for i in range(proposals.shape[0]):
        c = int(proposals[i,3])
        if c not in proposal_dict.keys():
            proposal_dict[c]=[]
        c_score = prop_iou[i, 0] + proposals[i, 2]
        proposal_dict[c].append([proposals[i, 0] / t_factor, proposals[i, 1] / t_factor, c, c_score])

    return  proposal_dict