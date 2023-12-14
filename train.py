import torch
from torch.utils import data
import random

def S_train(step, args, net, loader_iter, optimizer, logger):
    net.train()
    total_loss = {}
    total_cost = []
    optimizer.zero_grad()

    for batch in range(args.batch_size):
        sample = next(loader_iter)
        data, vid_label, point_label = sample['data'], sample['vid_label'], sample['point_label']
        data = data.to(args.device)
        vid_label = vid_label.to(args.device)
        point_label = point_label.to(args.device)

        outputs = net(data, vid_label)
        cost, loss_dict = net.criterion(args, outputs, vid_label, point_label)

        total_cost.append(cost)
        if not torch.isnan(cost):
            for key in loss_dict.keys():
                if not (key in total_loss.keys()):
                    total_loss[key] = []
                if loss_dict[key] > 0:
                    total_loss[key] += [loss_dict[key].detach().cpu().item()]
                else:
                    total_loss[key] += [loss_dict[key]]
                    
    total_cost = sum(total_cost) / args.batch_size
    total_cost.backward()
    optimizer.step()

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / args.batch_size, step)

    return total_cost.detach().cpu().item()


def I_train(epoch, args, train_dataset, net, optimizer, logger):
    net.train()
    loss_dict_sum = {}

    indices_train = list(range(len(train_dataset)))
    random.shuffle(indices_train)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices_train)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size2, num_workers=args.num_workers,
                                    sampler=sampler_train, drop_last=True, collate_fn=train_dataset.collate_fn)

    for sample in train_loader:
        features, proposals, pseudo_labels = sample['data'], sample['proposals'], sample['psuedo_label']
        features = [torch.from_numpy(feat).float().to(args.device) for feat in features]
        proposals = [torch.from_numpy(prop[:,:2]).float().to(args.device) for prop in proposals]
        pseudo_labels = [torch.from_numpy(pseudo_label).float().to(args.device) for pseudo_label in pseudo_labels]
        
        outputs = net(features, proposals, pseudo_labels)
        loss_dict = net.criterion(outputs, args=args)
        for key in loss_dict.keys():
            if key not in loss_dict_sum.keys():
                loss_dict_sum[key] = 0
            loss_dict_sum[key] += loss_dict[key].cpu().item()

        loss_total = loss_dict['loss_total']
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
    total_loss = loss_dict_sum['loss_total'] / len(train_loader)
    for key in loss_dict_sum.keys():
        logger.log_value("loss/" + key, loss_dict_sum[key] / len(train_loader), epoch)
    return total_loss





