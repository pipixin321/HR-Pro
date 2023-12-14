import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import utils

class Reliable_Memory(nn.Module):
    def __init__(self, num_class, feat_dim):
        super(Reliable_Memory, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.proto_momentum = 0.001 
        self.proto_num = 1
        self.proto_vectors = torch.nn.Parameter(torch.zeros([self.num_class, self.proto_num, self.feat_dim]), requires_grad=False)
        
    def init(self, args, net, train_loader):
        print('Memory initialization in progress...')
        with torch.no_grad():
            net.eval()
            pfeat_total = {}
            temp_loader = data.DataLoader(train_loader.dataset, batch_size=1, shuffle=False, num_workers=4)
            for sample in temp_loader:
                _data, vid_label, point_anno = sample['data'], sample['vid_label'], sample['point_label']
                outputs = net(_data.to(args.device), vid_label.to(args.device))
                embeded_feature = outputs['embeded_feature']
                for b in range(point_anno.shape[0]):
                    gt_class = torch.nonzero(vid_label[b]).squeeze(1).numpy()
                    for c in gt_class:
                        select_id = torch.nonzero(point_anno[b, :, c]).squeeze(1)
                        if select_id.shape[0] > 0:
                            act_feat = embeded_feature[b, select_id, :]
                            if c not in pfeat_total.keys():
                                pfeat_total[c] = act_feat
                            else:
                                pfeat_total[c] = torch.cat([pfeat_total[c], act_feat])

            for c in range(self.num_class):
                cluster_centers = pfeat_total[c].mean(dim=0, keepdim=True)
                self.proto_vectors[c] = cluster_centers


    def update(self, args, feats, act_seq, vid_label):
        self.proto_vectors = self.proto_vectors.to(args.device)
        feat_list = {}
        for b in range(act_seq.shape[0]):
            gt_class = torch.nonzero(vid_label[b]).cpu().squeeze(1).numpy()
            for c in gt_class:
                select_id = torch.nonzero(act_seq[b, :, c]).squeeze(1)
                if select_id.shape[0] > 0:
                    act_feat = feats[b, select_id, :]
                    if c not in feat_list.keys():
                        feat_list[c] = act_feat
                    else:
                        feat_list[c] = torch.cat(feat_list[c], act_feat)

        for c in feat_list.keys():
            if len(feat_list[c]) > 0:
                feat_update = feat_list[c].mean(dim=0, keepdim=True)
                self.proto_vectors[c] = (1 - self.proto_momentum) * self.proto_vectors[c] + self.proto_momentum * feat_update


class Reliabilty_Aware_Block(nn.Module):
    def __init__(self, input_dim, dropout, num_heads=8, dim_feedforward=128, pos_embed=False):
        super(Reliabilty_Aware_Block, self).__init__()
        self.conv_query = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_key = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_value = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)

        self.self_atten = nn.MultiheadAttention(input_dim, num_heads=num_heads, dropout=0.1)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, attn_mask=None,):
        src = features.permute(2, 0, 1)
        q = k = src
        q = self.conv_query(features).permute(2, 0, 1)
        k = self.conv_key(features).permute(2, 0, 1)

        src2, attn = self.self_atten(q, k, src, attn_mask=attn_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src.permute(1, 2, 0)
        return src, attn

  
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim

        RAB_args = args.RAB_args
        self.RAB = nn.ModuleList([
            Reliabilty_Aware_Block(
                input_dim=self.feature_dim,
                dropout=RAB_args['drop_out'],
                num_heads=RAB_args['num_heads'],
                dim_feedforward=RAB_args['dim_feedforward'])
            for i in range(RAB_args['layer_num'])
        ])

        self.feature_embedding = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, input_features, prototypes=None):
        '''
        input_feature: [B,T,F]
        prototypes：[C,1,F]
        '''
        B, T, F = input_features.shape
        input_features = input_features.permute(0, 2, 1)                        #[B,F,T]
        prototypes = prototypes.to(input_features.device)                       #[C,1,F]
        prototypes = prototypes.view(1,F,-1).expand(B,-1,-1)                    #[B,F,C]
        if hasattr(self, 'RAB'):
            layer_features = torch.cat([input_features, prototypes], dim=2)     #[B,F,T+C]
            for layer in self.RAB:
                layer_features, _ = layer(layer_features)
            input_features = layer_features[:, :, :T]                           #[B,F,T]
        embeded_features = self.feature_embedding(input_features)               #[B,F,T]

        return embeded_features
    

class S_Model(nn.Module):
    def __init__(self, args):
        super(S_Model, self).__init__()
        self.feature_dim = args.feature_dim
        self.num_class = args.num_class
        self.r_act = args.r_act
        self.dropout = args.dropout

        self.memory = Reliable_Memory(self.num_class, self.feature_dim)
        self.encoder = Encoder(args)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.num_class + 1, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.sigmoid = nn.Sigmoid()
        self.bce_criterion = nn.BCELoss(reduction='none')
        self.lambdas = args.lambdas

        


    def forward(self, input_features, vid_labels=None):
        '''
        input_feature: [B,T,F]
        '''
        # >> Encoder and classifier
        embeded_feature = self.encoder(input_features, self.memory.proto_vectors)   #[B,F,T]
        cas = self.classifier(embeded_feature)                                      #[B,C+1,T]
        cas = cas.permute(0, 2, 1)                                                  #[B,T,C+1]
        cas = self.sigmoid(cas)                                                     #[B,T,C+1]
        # class-Specific activation sequence
        cas_S = cas[:, :, :-1]                                                      #[B,T,C]
        # class-Agnostic attention sequence (background)
        bkg_score = cas[:, :, -1]                                                   #[B,T]

        # >> Fusion
        cas_P = cas_S * (1 - bkg_score.unsqueeze(2))                                #[B,T,C]
        cas_fuse = torch.cat((cas_P, bkg_score.unsqueeze(2)), dim=2)                #[B,T,C+1]

        # >> Top-k pooling
        value, _ = cas_S.sort(descending=True, dim=1)
        k_act = max(1, input_features.shape[1] // self.r_act)
        topk_scores = value[:, :k_act, :]
        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + \
                        (torch.mean(cas_S, dim=1) * (1 - vid_labels))

        return dict(
            cas_fuse = cas_fuse,                                                    #[B,T,C+1]
            cas_S = cas_S,                                                          #[B,T,C+1]   
            vid_score = vid_score,                                                  #[B,C]
            embeded_feature = embeded_feature.permute(0, 2, 1),                     #[B,T,F]                   
        )

    def criterion(self, args, outputs, vid_label, point_label):
        vid_score, embeded_feature, cas_fuse = outputs['vid_score'], outputs['embeded_feature'], outputs['cas_fuse']
        point_label = torch.cat((point_label, torch.zeros((point_label.shape[0], point_label.shape[1], 1)).to(args.device)), dim=2)
        act_seed, bkg_seed = utils.select_seed(cas_fuse[:, :, -1].detach().cpu(), point_label.detach().cpu())

        loss_dict = {}
        # >> base loss
        loss_vid, loss_frame, loss_frame_bkg = self.base_loss_func(args, act_seed, bkg_seed, vid_score, vid_label, cas_fuse, point_label)
        loss_dict["loss_vid"] = loss_vid
        loss_dict["loss_frame"] = loss_frame
        loss_dict["loss_frame_bkg"] = loss_frame_bkg

        # >> feat loss
        loss_contrastive = self.feat_loss_func(args, embeded_feature, act_seed, bkg_seed, vid_label)
        loss_dict["loss_contrastive"] = loss_contrastive

        # >> update memory
        self.memory.update(args, embeded_feature.detach(), act_seed, vid_label)

        loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame \
                    + self.lambdas[2] * loss_frame_bkg + self.lambdas[3] * loss_contrastive
        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict


    def base_loss_func(self, args, act_seed, bkg_seed, vid_score, vid_label, cas_sigmoid_fuse, point_anno):
        # >> video-level loss
        loss_vid = self.bce_criterion(vid_score, vid_label)
        loss_vid = loss_vid.mean()

        # >> frame-level loss
        loss_frame = 0
        loss_frame_bkg = 0
        # act frame loss
        act_seed = act_seed.to(args.device)
        focal_weight_act = (1 - cas_sigmoid_fuse ) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)
        loss_frame = (((focal_weight_act * self.bce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act)
                        .sum(dim=2)).sum(dim=1) / (num_actions + 1e-6)).mean()
        # bkg frame loss
        bkg_seed = bkg_seed.unsqueeze(-1).to(args.device)
        point_anno_bkg = torch.zeros_like(point_anno).to(args.device)
        point_anno_bkg[:, :, -1] = 1
        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1).squeeze(1)
        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        loss_frame_bkg = (((focal_weight_bkg * self.bce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg)
                            .sum(dim=2)).sum(dim=1) / (num_bkg + 1e-6)).mean()

        return loss_vid, loss_frame, loss_frame_bkg
    
    def feat_loss_func(self, args, embeded_feature, act_seed, bkg_seed, vid_label):
        loss_contra = 0
        proto_vectors = utils.norm(self.memory.proto_vectors.to(args.device))                                        #[C,N,F]                                                             
        for b in range(act_seed.shape[0]):
            # >> extract pseudo-action/background features
            gt_class = torch.nonzero(vid_label[b]).squeeze(1)
            act_feat_lst = []
            for c in gt_class:
                act_feat_lst.append(utils.extract_region_feat(act_seed[b, :, c], embeded_feature[b, :, :]))
            bkg_feat = utils.extract_region_feat(bkg_seed[b].squeeze(-1), embeded_feature[b, :, :])
            
            # >> caculate similarity matrix
            if len(bkg_feat) == 0:
                continue
            bkg_feat = utils.norm(torch.cat(bkg_feat, 0))                                                            #[t_b,F]
            b_sim_matrix = torch.matmul(bkg_feat.unsqueeze(0).expand(args.num_class, -1, -1), 
                                        torch.transpose(proto_vectors, 1, 2)) / 0.1                                  #[C,t_b,N]
            b_sim_matrix = torch.exp(b_sim_matrix).reshape(b_sim_matrix.shape[0], -1).mean(dim=-1)                   #[C]
            for idx, act_feat in enumerate(act_feat_lst):
                if act_feat is not None:
                    if len(act_feat) == 0:
                        continue
                    act_feat = utils.norm(torch.cat(act_feat, 0))                                                    #[t_a,F]
                    a_sim_matrix = torch.matmul(act_feat.unsqueeze(0).expand(args.num_class, -1, -1), 
                                                torch.transpose(proto_vectors, 1, 2)) / 0.1                          #[C,t_a,N]
                    a_sim_matrix = torch.exp(a_sim_matrix).reshape(a_sim_matrix.shape[0], -1).mean(dim=-1)           #[C]                                                      

            # >> caculate contrastive loss
                    c = gt_class[idx]
                    loss_contra_act = - torch.log(a_sim_matrix[c] / a_sim_matrix.sum())
                    loss_contra_bkg = - torch.log(a_sim_matrix[c] / 
                                                 (a_sim_matrix[c] + b_sim_matrix[c]))
                    loss_contra += (0.5 * loss_contra_act + 0.5 * loss_contra_bkg)

            loss_contra = loss_contra / gt_class.shape[0]
        loss_contra = loss_contra / act_seed.shape[0]

        return loss_contra
        



