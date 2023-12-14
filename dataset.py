import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random


class dataset(Dataset):
    def __init__(self, args, phase="train", sample="random", stage=1):
        self.args = args
        self.phase = phase
        self.sample = sample
        self.stage = stage
        self.num_segments = args.num_segments
        self.class_name_lst = args.class_name_lst
        self.class_idx_dict = {cls: idx for idx, cls in enumerate(self.class_name_lst)}
        self.num_class = args.num_class
        self.t_factor = args.frames_per_sec / args.segment_frames_num
        self.data_path = args.data_path
        self.feature_dir = os.path.join(self.data_path, 'features', self.phase)
        self._prepare_data()
    
    def _prepare_data(self):
        # >> video list
        self.data_list = [item.strip() for item in list(open(os.path.join(self.data_path, "split_{}.txt".format(self.phase))))]
        print("number of {} videos:{}".format(self.phase, len(self.data_list)))
        with open(os.path.join(self.data_path, "gt_full.json")) as f:
            self.gt_dict = json.load(f)["database"]

        # >> video label
        self.vid_labels = {}
        for item_name in self.data_list:
            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.num_class)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.class_idx_dict[ann_label]] = 1.0
            self.vid_labels[item_name] = item_label

        # >> point label
        self.point_anno = pd.read_csv(os.path.join(self.data_path, 'point_labels', 'point_gaussian.csv'))

        if self.stage == 2:
            with open(os.path.join(self.args.output_path_s1, 'proposals.json'.format(self.phase)),'r') as f:
                self.proposals_json = json.load(f)
            self.load_proposals()

        # >> ambilist
        if self.args.dataset == "THUMOS14":
            ambilist = './dataset/THUMOS14/Ambiguous_test.txt'
            ambilist = list(open(ambilist, "r"))
            self.ambilist = [a.strip("\n").split(" ") for a in ambilist]
      

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vid_name = self.data_list[idx]
        vid_feature = np.load(os.path.join(self.feature_dir, vid_name + ".npy"))
        data, vid_len, sample_idx = self.process_feat(vid_feature)
        vid_label, point_label, vid_duration = self.process_label(vid_name, vid_len, sample_idx)
        if self.stage == 1:
            sample = dict(
                data = data, 
                vid_label = vid_label, 
                point_label = point_label,
                vid_name = vid_name, 
                vid_len = vid_len, 
                vid_duration = vid_duration,
            )
            return sample

        elif self.stage == 2 :
            p_prop = self.proposals_new[vid_name]['PP']
            if self.phase == 'test':
                return dict(
                    vid_name = vid_name, data = data, vid_label = vid_label, point_label = point_label, vid_duration = vid_duration,
                    proposals = p_prop,
                )
            else:
                r_prop = self.proposals_new[vid_name]['RP']
                n_prop = self.proposals_new[vid_name]['NP']
                proposals = np.concatenate((p_prop[:, :3], n_prop), axis=0)
                if proposals.shape[0] > self.args.max_proposal:
                    proposals =proposals[random.sample(list(range(proposals.shape[0])), self.args.max_proposal), :]
                return dict(
                    vid_name = vid_name, data = data, vid_label = vid_label, point_label = point_label, vid_duration = vid_duration,
                    proposals = proposals, psuedo_label = r_prop, pseudo_bkg = n_prop,
                )
                    
    def process_feat(self, vid_feature):
        vid_len = vid_feature.shape[0]
        if vid_len <= self.num_segments or self.num_segments == -1:
            sample_idx = np.arange(vid_len).astype(int)
        elif self.num_segments > 0 and self.sample == "random":
            sample_idx = np.arange(self.num_segments) * vid_len / self.num_segments
            for i in range(self.num_segments):
                if i < self.num_segments - 1:
                    if int(sample_idx[i]) != int(sample_idx[i + 1]):
                        sample_idx[i] = np.random.choice(range(int(sample_idx[i]), int(sample_idx[i + 1]) + 1))
                    else:
                        sample_idx[i] = int(sample_idx[i])
                else:
                    if int(sample_idx[i]) < vid_len - 1:
                        sample_idx[i] = np.random.choice(range(int(sample_idx[i]), vid_len))
                    else:
                        sample_idx[i] = int(sample_idx[i])
        elif self.num_segments > 0 and self.sample == 'uniform':
            samples = np.arange(self.num_segments) * vid_len / self.num_segments
            samples = np.floor(samples)
            sample_idx =  samples.astype(int)
        else:
            raise AssertionError('Not supported sampling !')
        feature = vid_feature[sample_idx]
        
        return feature, vid_len, sample_idx

    def process_label(self, vid_name, vid_len, sample_idx):
        vid_label = self.vid_labels[vid_name]
        vid_duration, vid_fps = self.gt_dict[vid_name]['duration'], self.gt_dict[vid_name]['fps']

        if self.num_segments == -1:
            self.t_factor_point = self.args.frames_per_sec / (vid_fps * 16)
            temp_anno = np.zeros([vid_len, self.num_class], dtype=np.float32)
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class']]
            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = self.class_idx_dict[temp_df['class'][key]]
                temp_anno[int(point * self.t_factor_point)][class_idx] = 1
            point_label = temp_anno[sample_idx, :]
            return vid_label, point_label, vid_duration
        
        else:
            self.t_factor_point = self.num_segments / (vid_fps * vid_duration)
            temp_anno = np.zeros([self.num_segments, self.num_class], dtype=np.float32)
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class']]
            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = self.class_idx_dict[temp_df['class'][key]]
                temp_anno[int(point * self.t_factor_point)][class_idx] = 1
            point_label = temp_anno
            return vid_label, point_label, vid_duration

    def load_proposals(self):
        proposals = self.proposals_json[self.phase]
        self.proposals_new = dict()
        for vid_name in self.data_list:
            self.proposals_new[vid_name] = dict()
            # >> caculate t_factor(time --> snippet)
            t_factor = self.t_factor

            # >> load Positive Proposals
            PP = proposals[vid_name]['PP']
            PP_new = []
            for label in PP.keys():
                for prop in PP[label]:
                    prop_new = [round(prop[0]*t_factor, 3), round(prop[1]*t_factor, 3), prop[2], self.class_idx_dict[label]]
                    PP_new.append(prop_new)
            if len(PP_new) == 0:
                PP_new.append([0, 0, 0])
            self.proposals_new[vid_name]['PP'] = np.array(PP_new)

            if self.phase == 'train':
                # >> load Reliable Proposals
                RP = proposals[vid_name]['RP']
                RP_new = []
                for label in RP.keys():
                    for prop in RP[label]:
                        if prop is not None:
                            prop_new = [round(prop[0]*t_factor, 3), round(prop[1]*t_factor, 3), prop[2]]
                            RP_new.append(prop_new)
                if len(RP_new) == 0:
                    RP_new.append([0, 0, 0])
                self.proposals_new[vid_name]['RP'] = np.array(RP_new)

                # >> load Negative Proposals
                NP = proposals[vid_name]['NP']
                self.proposals_new[vid_name]['NP'] = np.array(NP)

    def collate_fn(self, batch):
        """
        Collate function for creating batches of data samples.
        """
        keys = batch[0].keys()
        data = {key: [] for key in keys}
        for sample in batch:
            for key in keys:
                data[key].append(sample[key])
        return data