import os
import json
from tqdm import tqdm
import torch
import numpy as np
import random
from utils import select_seed, grouping
from log import color

def reliability_ranking(args, train_loader, test_loader):
    '''
    point-based proposal generation
    '''
    proposals_dict = dict()
    for subset in ['test', 'train']:
        snippet_result_path = os.path.join(args.output_path_s1, 'snippet_result_{}.json'.format(subset))
        assert os.path.exists(snippet_result_path)
        with open(snippet_result_path, 'r') as json_file:
            snippet_result = json.load(json_file)

        data_loader = train_loader if subset == 'train' else test_loader 
        if subset == 'train':
            point_anno = data_loader.dataset.point_anno

        sub_proposals_dict = dict()
        for sample in tqdm(data_loader):
            data, vid_name, vid_duration = sample['data'], sample['vid_name'][0], sample['vid_duration'][0]
            sub_proposals_dict[vid_name] = dict()
            # >> Positive Proposals
            PP = {}
            for pred in snippet_result['results'][vid_name]:
                label, segment, score = pred['label'], pred['segment'], pred['score']
                if label not in PP.keys():
                    PP[label] = []
                PP[label].append([segment[0], segment[1], score])
            sub_proposals_dict[vid_name]['PP'] = PP

            if subset == 'train':
                point_label = sample['point_label']
                vid_fps = data_loader.dataset.gt_dict[vid_name]['fps']
                # >> Reliable Proposals
                points_df = point_anno[point_anno["video_id"] == vid_name][['point', 'class']].sort_values(by='point', ascending=True)
                points = []
                for key in points_df['point'].keys():
                    t_point = points_df['point'][key] / vid_fps                # frame -> time
                    points.append([t_point, points_df['class'][key]])
                RP = match_point(PP, points)
                sub_proposals_dict[vid_name]['RP'] = RP

                # >> Negative Proposals
                bkg_score = np.asarray(snippet_result['bkg_score'][vid_name], dtype=np.float32)
                _, bkg_seed = select_seed(torch.from_numpy(bkg_score).unsqueeze(0), point_label)
                segs = grouping(np.where(bkg_seed[0] > 0)[0])
                NP = []
                num_PP = sum([len(PP[c]) for c in PP.keys()])
                num_NP = int(0.5 * num_PP)
                if len(segs) > 0:
                    while len(NP) < num_NP:
                        seg_idx = random.randint(0, len(segs)-1)
                        seg = list(segs[seg_idx])
                        if len(seg) == 1:
                            NP.append([float(seg[0]), float(seg[0]+1), -1])
                        else:
                            start = int(random.uniform(seg[0], seg[-1]))
                            end = int(random.uniform(start, seg[-1]))
                            if start < end:
                                NP.append([float(start), float(end), -(end-start)/(seg[-1]-seg[0])])
                sub_proposals_dict[vid_name]['NP'] = NP
        proposals_dict[subset] = sub_proposals_dict

    json_path = os.path.join(args.output_path_s1, 'proposals.json')
    with open(json_path, 'w') as f:
        json.dump(proposals_dict, f)
    print(color('>> Reliability-aware Ranking Finished'))

def match_point(props, points):
    '''
    assign each point to the proposal that has the highest confidence
    '''
    t_pos = [-1] + [point[0] for point in points] + [1e5]
    RP = {}
    for i, point in enumerate(points):
        t, label = point
        p_match, max_score = None, 0
        # with boundary condition
        if label in props.keys():
            for prop in props[label]:
                inclusive_condition = (prop[0] <= t and prop[1] >= t)
                boundary_condition = (prop[0] > t_pos[i] and prop[1] < t_pos[i+2])
                score_condition = prop[2] > max_score
                if inclusive_condition and boundary_condition and score_condition:
                    p_match, max_score = prop, prop[2]
        # without boundary condition
        if p_match is None and label in props.keys():
            for prop in props[label]:
                inclusive_condition = (prop[0] <= t and prop[1] >= t)
                score_condition = prop[2] > max_score
                if inclusive_condition and score_condition:
                    p_match, max_score = prop, prop[2]

        if label not in RP.keys():
            RP[label] = []
        RP[label].append(p_match)
                    
    return RP  
