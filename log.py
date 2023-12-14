import os
import numpy as np
import json
from termcolor import colored
from eval.eval_detection import ANETdetection


def color(text, txt_color='green', attrs=['bold']):
    return colored(text, txt_color, attrs=attrs)

def save_config(config, file_path):
    config = vars(config)
    for k, v in config.items():
        if type(v) is np.ndarray:
            config[k] = list(v)
    with open(file_path, "w") as fo:
        fo.write(json.dumps(config, indent=4))

def initial_log(log_filepath, args):
    task_descr = """
        {title}
            - dataset:\t {dataset}
            - optimization stage: {stage}
            - description: {descr}
            - device: {device}
        """.format(
            title=color('Temporal Action Localization', 'magenta'),
            dataset=color(args.dataset, 'white', attrs=['bold', 'underline']),
            stage = args.stage,
            descr=args.task_info,
            device=args.device,
        )
    print(task_descr)

    if os.path.exists(log_filepath):
        os.remove(log_filepath)
    with open(log_filepath, 'w') as f:
        f.write('\n{sep}\n{info}\n{sep}\n'.format(sep = '*' * 10, info=task_descr))
        title = '| {:^6s} | {:^8s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^15} | {:^15} | {:^15} | {:^15} | {:^20} |'.format(
            'step', 'test_acc', 'loss', '@0.1', '@0.2', '@0.3', '@0.4', '@0.5', '@0.6', '@0.7', 'avg@(0.1:0.5)', 'avg@(0.3:0.7)', 'avg@(0.1:0.7)', 'Elapsed time', 'Now')
        f.write('+{sep}+\n'.format(sep = '-'*(len(title)-2)))
        f.write('{}\n'.format(title))
        f.write('{sep}\n'.format(sep = '-'*len(title)))


def save_best_record(test_info, log_filepath):
    with open(log_filepath, 'a') as f:
        f.write('| {:^6s} | {:^8s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^6s} | {:^15} | {:^15} | {:^15} | {:^15} | {:^20} | \n'.format(
            str(test_info["step"][-1]), '{:.2f}'.format(100*test_info['test_acc'][-1]), '{:.3f}'.format(test_info['loss'][-1]),
            '{:.2f}'.format(100*test_info["mAP@0.1"][-1]), '{:.2f}'.format(100*test_info["mAP@0.2"][-1]), '{:.2f}'.format(100*test_info["mAP@0.3"][-1]),
            '{:.2f}'.format(100*test_info["mAP@0.4"][-1]), '{:.2f}'.format(100*test_info["mAP@0.5"][-1]), '{:.2f}'.format(100*test_info["mAP@0.6"][-1]), 
            '{:.2f}'.format(100*test_info["mAP@0.7"][-1]),
            '{:.2f}'.format(100*test_info["average_mAP[0.1:0.5]"][-1]),
            '{:.2f}'.format(100*test_info["average_mAP[0.3:0.7]"][-1]),
            '{:.2f}'.format(100*test_info["average_mAP[0.1:0.7]"][-1]),
            test_info['elapsed'][-1],
            test_info['now'][-1],
        ))
def log_evaluate(args, step, test_acc, logger, json_path, test_info, subset='test'):
    # >> evaluate mAP
    mapping_subset = {'THUMOS14':{'train':'Validation', 'test':'Test'}}
    subset_name = mapping_subset[args.dataset][subset]
    gt_path = os.path.join(args.data_path, "gt_full.json")
    anet_detection = ANETdetection(gt_path, json_path, subset=subset_name, tiou_thresholds=args.tIoU_thresh,
                                    verbose=False, check_status=False, blocked_videos=args.blocked_videos)
    mAP, _ = anet_detection.evaluate()

    # >> log mAP
    log_folder = 'acc'
    test_info['step'].append(step)
    test_info['test_acc'].append(test_acc)
    if logger is not None:
        logger.log_value('{}/Test accuracy'.format(log_folder), test_acc, step)

    test_info["average_mAP[0.1:0.7]"].append(mAP[:7].mean())
    test_info["average_mAP[0.1:0.5]"].append(mAP[:5].mean())
    test_info["average_mAP[0.3:0.7]"].append(mAP[2:7].mean())
    for i in range(len(args.tIoU_thresh)):
        test_info["mAP@{:.1f}".format(args.tIoU_thresh[i])].append(mAP[i])
    
    if logger is not None:
        logger.log_value('{}/average mAP[0.1:0.7]'.format(log_folder), mAP[:7].mean(), step)
        logger.log_value('{}/average mAP[0.1:0.5]'.format(log_folder), mAP[:5].mean(), step)
        logger.log_value('{}/average mAP[0.3:0.7]'.format(log_folder), mAP[2:7].mean(), step)
        for i in range(len(args.tIoU_thresh)):
            logger.log_value('{}/mAP@{:.1f}'.format(log_folder, args.tIoU_thresh[i]), mAP[i], step)

        return test_info["average_mAP[0.1:0.7]"][-1]   
    
