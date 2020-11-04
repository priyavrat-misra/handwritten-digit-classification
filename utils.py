import torch
from collections import namedtuple
from itertools import product

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images = batch[0].to(device)
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds
