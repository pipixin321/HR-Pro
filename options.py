import os
import argparse
import yaml
import numpy as np

_CLASS_NAME = {
    "THUMOS14": [
        'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
        'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
        'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow',
        'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing',
        'ThrowDiscus', 'VolleyballSpiking'
    ],
}

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def parse_args():
    parser = argparse.ArgumentParser("Official Pytorch Implementation of HR-Pro: Point-supervised Temporal Action Localization \
                                        via Hierarchical Reliability Propagation")
    
    parser.add_argument('--cfg', type=str, default='thumos', help='hyperparameters path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'infer'])
    parser.add_argument('--stage', type=int, default=1, help='traning stage', choices=[1, 2])
    parser.add_argument('--seed', type=int, default=0, help='random seed (-1 for no manual seed)')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='root folder for saving models,ouputs,logs')
    args = parser.parse_args()

    # hyper-params from ymal file
    with open('./cfgs/{}_hyp.yaml'.format(args.cfg)) as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in hyp_dict.items():
        setattr(args, key, value)

    return init_args(args)

def init_args(args):
    # create folder for models/outputs/logs of stage1/stage2
    args.root_s1 = os.path.join(args.ckpt_path, args.dataset, args.task_info, 'stage1')
    args.model_path_s1 = os.path.join(args.root_s1, 'models' )
    args.output_path_s1 = os.path.join(args.root_s1, "outputs")
    args.log_path_s1 = os.path.join(args.root_s1, "logs")

    args.root_s2 = os.path.join(args.ckpt_path, args.dataset, args.task_info, 'stage2')
    args.model_path_s2 = os.path.join(args.root_s2, 'models' )
    args.output_path_s2 = os.path.join(args.root_s2, "outputs")
    args.log_path_s2 = os.path.join(args.root_s2, "logs")

    for dir in [args.model_path_s1, args.log_path_s1, args.output_path_s1,
                args.model_path_s2, args.log_path_s2, args.output_path_s2]:
        mkdir(dir)

    # mapping parameters of string format
    args.act_thresh_cas = eval(args.act_thresh_cas)
    args.act_thresh_agnostic = eval(args.act_thresh_agnostic)
    args.lambdas = eval(args.lambdas)
    args.tIoU_thresh = eval(args.tIoU_thresh)
    
    # get list of class name 
    args.class_name_lst = _CLASS_NAME[args.dataset]
    args.num_class = len(args.class_name_lst)

    # define format of test information
    args.test_info = {
        "step": [], "test_acc": [], 'loss':[], 'elapsed': [], 'now': [],
        "average_mAP[0.1:0.7]": [], "average_mAP[0.1:0.5]": [], "average_mAP[0.3:0.7]": [],
        "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], "mAP@0.4": [], 
        "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []
    }

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
