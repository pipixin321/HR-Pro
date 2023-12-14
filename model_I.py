import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Boundary_Completeness_Regressor(torch.nn.Module):
    def __init__(self, feat_dim, dropout_ratio, roi_size):
        super().__init__()
        self.feat_dim = feat_dim
        self.roi_size = roi_size
        self.hidden_dim_1d = self.feat_dim // 2

        self.start_sample_num = roi_size // 3
        self.end_sample_num = roi_size // 3

        self.start_reg_conv = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 2, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim // 2, 1, 1),
        )

        self.end_reg_conv = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 2, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim // 2, 1, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.prop_fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.prop_completeness = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 2, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim // 2, 1, 1),
        )

    def forward(self, feat, proposals, mask, pseudo_labels=None, iou_thresh=0.5, is_training=True):
        mask = mask.bool()

        feat1 = feat[:, :,                   : self.roi_size//6  , :].max(2)[0]
        feat2 = feat[:, :, self.roi_size//6  : self.roi_size//6*5, :].max(2)[0]
        feat3 = feat[:, :, self.roi_size//6*5:                   , :].max(2)[0]
        iou_feat = torch.cat((feat2 - feat1, feat2, feat2 - feat3), dim=2)

        feat_fuse = self.prop_fusion(iou_feat)                          #[B,M,D]
        feat_fuse = feat_fuse.transpose(-1, -2)                         #[B,D,M]
        prop_iou = self.prop_completeness(feat_fuse).squeeze(1)         #[B,M]
        iou_pred = self.sigmoid(prop_iou)

        feat1 = feat1.transpose(-1, -2)                                 #[B,F,M]
        feat3 = feat3.transpose(-1, -2)                                 #[B,F,M]
        start_reg = self.start_reg_conv(feat1).squeeze(1)               #[B,M]
        end_reg = self.end_reg_conv(feat3).squeeze(1)                   #[B,M]

        reg_loss_b = 0.0
        iou_loss_b = 0.0
        if is_training:
            proposal_refined_b = []
            for b in range(len(proposals)):
                proposal_refined = self.refine_proposals_se(proposals[b],mask[b],start_reg[b],end_reg[b])
                reg_loss, iou_gt = self.regression_loss(proposal_refined, proposals[b], pseudo_labels[b], iou_thresh)
                iou_loss = self.iou_loss(iou_pred[b],mask[b],iou_gt,iou_thresh)
                proposal_refined_b.append(proposal_refined)
                reg_loss_b = reg_loss_b + reg_loss
                iou_loss_b = iou_loss_b + iou_loss
            reg_loss_b = reg_loss_b / len(proposals)
            iou_loss_b = iou_loss_b / len(proposals)

            return proposal_refined_b, iou_pred, reg_loss_b, iou_loss_b

        else:
            proposal_refined_b = []
            for b in range(len(proposals)):
                proposal_refined = self.refine_proposals_se(proposals[b],mask[b],start_reg[b],end_reg[b])
                proposal_refined_b.append(proposal_refined)
            return proposal_refined_b, iou_pred, reg_loss_b, iou_loss_b


    def refine_proposals_se(self, proposals, mask, start_reg, end_reg):
        widths =  proposals[:,1] - proposals[:,0]
        s_offset = start_reg[mask] * widths
        e_offset = end_reg[mask] * widths
        pred_props = proposals.clone()
        pred_props[:,0] = proposals[:,0] + s_offset
        pred_props[:,1] = proposals[:,1] + e_offset
        return pred_props
    
    def segments_iou(self, segments1, segments2):
        segments1 = segments1.unsqueeze(1)                          # [M1, 1, 2]
        segments2 = segments2.unsqueeze(0)                          # [1, M2, 2]
        tt1 = torch.maximum(segments1[..., 0], segments2[..., 0])   # [M1, M2]
        tt2 = torch.minimum(segments1[..., 1], segments2[..., 1])   # [M1, M2]
        intersection = tt2 - tt1
        union = (segments1[..., 1] - segments1[..., 0]) + (segments2[..., 1] - segments2[..., 0]) - intersection
        iou = intersection / (union + 1e-6)                         # [M1, M2]
        # Remove negative values
        iou_temp = torch.zeros_like(iou)
        iou_temp[iou > 0] = iou[iou > 0]
        return iou_temp
    
    def regression_loss(self, proposal_refined, proposals, pseudo_label, iou_thresh):
        iou = self.segments_iou(proposals,pseudo_label)
        iou_max, idx = torch.max(iou,dim=1)
        
        refined_iou = self.segments_iou(proposal_refined,pseudo_label)
        refined_iou, idx = torch.max(refined_iou,dim=1)
        reg_loss = F.smooth_l1_loss(refined_iou, torch.ones(refined_iou.shape[0]).to(refined_iou.device), reduction='none')
        weight = (iou_max >= iou_thresh).float()
        if torch.sum(weight) > 0:
            reg_loss = torch.sum(reg_loss * weight) / torch.sum(weight)
        else:
            reg_loss = torch.sum(reg_loss * weight)
        return reg_loss,iou_max


    def iou_loss(self, iou_pred, mask, iou_gt, iou_thresh):
        u_hmask = (iou_gt >= iou_thresh).float()
        u_lmask = (iou_gt < iou_thresh).float()

        num_h = torch.sum(u_hmask)
        num_l = torch.sum(u_lmask)

        r_l = num_h / (num_l)
        device = iou_pred.device
        r_l = torch.min(r_l, torch.Tensor([1.0]).to(device))[0]
        u_slmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).to(device)
        u_slmask = u_slmask * u_lmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        iou_weights = u_hmask + u_slmask
        iou_loss = F.smooth_l1_loss(iou_pred[mask], iou_gt, reduction='none')
        if torch.sum(iou_weights) > 0:
            iou_loss = torch.sum(iou_loss * iou_weights) / torch.sum(iou_weights)
        else:
            iou_loss = torch.sum(iou_loss * iou_weights)
        return iou_loss

        
class I_Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        dropout_ratio = args.dropout
        self.feat_dim = args.feature_dim
        self.max_proposal = args.max_proposal
        self.roi_size = 12
        self.bcr = Boundary_Completeness_Regressor(self.feat_dim, dropout_ratio, self.roi_size)

    def forward(self, features, proposals, pseudo_labels=None, is_training=True):
        prop_features, prop_mask = self.extract_roi_features(features, proposals, is_training)
        prop_refined, iou_pred_orig, reg_loss, iou_loss = self.bcr(prop_features, proposals, prop_mask, 
                                                                    pseudo_labels, 0.3, is_training)
        prop_refined_feature, prop_mask1 = self.extract_roi_features(features, prop_refined, is_training)
        _, iou_pred_refined, _, _ = self.bcr(prop_refined_feature, prop_refined, prop_mask1, 
                                        pseudo_labels, 0.3, is_training=False)

        outputs = {
                'prop_refined': prop_refined,                           # [B, N, 2]
                'iou_pred_orig': iou_pred_orig.unsqueeze(-1),           # [B, M, 1]
                'iou_pred_refined': iou_pred_refined.unsqueeze(-1),     # [B, M, 1]
                'prop_mask': prop_mask,                                 # [B, M]
            }
        if is_training:
            outputs['reg_loss'] = reg_loss
            outputs['iou_loss'] = iou_loss
        return outputs
    
    def criterion(self, outputs, args):
        reg_loss, iou_loss = outputs['reg_loss'], outputs['iou_loss']
        prop_mask = outputs['prop_mask']
        prop_mask = prop_mask.unsqueeze(2).bool()                       # [B, M, 1]

        loss_prop_reg = args.weight_loss_reg * reg_loss 
        loss_prop_iou = args.weight_loss_score * iou_loss 
        loss_total = loss_prop_reg + loss_prop_iou 

        loss_dict = {
            'loss_total': loss_total,
            'loss_prop_reg': loss_prop_reg,
            'loss_prop_iou': loss_prop_iou,
        }
        return loss_dict

    def extract_roi_features(self, features, proposals, is_training):
        """
        Extract region of interest (RoI) features from raw i3d features based on given proposals

        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool

        Outputs:
            prop_features:tensor of size [B, M, roi_size, D]
            prop_mask: tensor of size [B, M]
        """
        num_prop = torch.tensor([prop.shape[0] for prop in proposals])
        batch, max_num = len(proposals), num_prop.max()
        # Limit the max number of proposals during training
        if is_training:
            max_num = min(max_num, self.max_proposal)
        prop_features = torch.zeros((batch, max_num, self.roi_size, self.feat_dim)).to(features[0].device)
        prop_mask = torch.zeros((batch, max_num)).to(features[0].device)

        for i in range(batch):
            feature = features[i]
            proposal = proposals[i]
            if num_prop[i] > max_num:
                sampled_idx = torch.randperm(num_prop[i])[:max_num]
                proposal = proposal[sampled_idx]

            # Extend the proposal by 25% of its length at both sides
            start, end = proposal[:, 0], proposal[:, 1]
            len_prop = end - start
            start_ext = start - 0.25 * len_prop
            end_ext = end + 0.25 * len_prop
            # Fill in blank at edge of the feature, offset 0.5, for more accurate RoI_Align results
            fill_len = torch.ceil(0.25 * len_prop.max()).long() + 1                         # +1 because of offset 0.5
            fill_blank = torch.zeros(fill_len, self.feat_dim).to(feature.device)
            feature = torch.cat([fill_blank, feature, fill_blank], dim=0)
            start_ext = start_ext + fill_len - 0.5
            end_ext = end_ext + fill_len - 0.5
            proposal_ext = torch.stack((start_ext, end_ext), dim=1)
            
            # Extract RoI features using RoI Align operation
            y1, y2 = proposal_ext[:, 0], proposal_ext[:, 1]
            x1, x2 = torch.zeros_like(y1), torch.ones_like(y2)
            boxes = torch.stack((x1, y1, x2, y2), dim=1)                                    # [M, 4]
            feature = feature.transpose(0, 1).unsqueeze(0).unsqueeze(3)                     # [1, D, T, 1]
            feat_roi = torchvision.ops.roi_align(feature, [boxes], [self.roi_size, 1])      # [M, D, roi_size, 1]
            feat_roi = feat_roi.squeeze(3).transpose(1, 2)                                  # [M, roi_size, D]
            prop_features[i, :proposal.shape[0], :, :] = feat_roi                           # [B, M, roi_size, D]
            prop_mask[i, :proposal.shape[0]] = 1                                            # [B, M]

        return prop_features, prop_mask
    
    
    
