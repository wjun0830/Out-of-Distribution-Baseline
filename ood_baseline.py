#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
from pprint import pprint
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import argparse
import datasets.datasets as datasets
from datasets.ood_datasets import get_dataset, get_superclass_list, get_subclass_dataset
from torch.utils.data import DataLoader
from resnet import ResNet34, ResNet18
import copy
import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import transforms
import time
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for [default: 10]')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--iter', default='False', type=str)
parser.add_argument('--dataset', type=str, default='cifar10', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--out-dataset', type=str, default='svhn', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--r_ratio', default=0.5, type=float, help='self-supervised loss ratio')
parser.add_argument('--workers', default=6, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--model', type=str, default='classifier32', help='classifier32 | WRN')
parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                    default=None, nargs="*", type=str)
parser.add_argument('--trial', default=0, type=int)
args = parser.parse_args()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
def kl_div(d1, d2):
    """
    Compute KL-Divergence between d1 and d2.
    """
    dirty_logs = d1 * torch.log2(d1 / d2)
    return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)), axis=1)

def train(epoch,net,optimizer,train_loader, iter):
    net.train()
    for batch_idx, (inputs_x, labels_x) in enumerate(train_loader):
        batch_size = inputs_x.size(0)
        inputs, labels_x = inputs_x.cuda(), labels_x.cuda()
        logits = net(inputs)
        loss = criterion(logits, labels_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write('\r')

        sys.stdout.write('%s: | Epoch [%3d/%3d] loss_x: %.2f, gr_loss: %.2f'
                         % (args.dataset, epoch, args.num_epochs, loss_x.item(), rotloss.item()))
        sys.stdout.flush()

    return iter

def test(net1, d):
    net1.eval()
    prediction1 = torch.zeros(50000)
    labels = torch.zeros(50000)
    total = 0
    total2 = 0
    open_labels = torch.ones(50000)

    probs = torch.zeros(50000)

    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(testloader):
            bsz = inputs.size(0)
            inputs, label = inputs.cuda(), label.long().cuda()

            o1 = net1(inputs)
            _1, pred1 = torch.softmax(o1, dim=1).max(1)

            total += bsz
            total2 += bsz
            for b in range(bsz):
                probs[n] = _1[b]
                labels[n] = label[b]
                prediction1[n] = pred1[b]
                n+=1

    # openset test
    open_labels[total:] = 0
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(ood_test_loader[d]):
            bsz = inputs.size(0)
            inputs, label = inputs.cuda(), label.long().cuda()

            o1 = net1(inputs)
            _1, pred1 = torch.softmax(o1, dim=1).max(1)
            total2 += bsz
            for b in range(bsz):
                probs[n] = _1[b]
                labels[n] = label[b]
                n += 1
    prob = probs[:total2].reshape(-1, 1)

    auc = roc_auc_score(open_labels[:total2].cpu().numpy(), prob)
    correct1 = prediction1[:total].eq(labels[:total]).sum().item()
    acc1 = correct1 / total
    return acc1, auc

acc_results = []
auc_results = []
criterion = nn.CrossEntropyLoss()
options = vars(args)
options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
if not os.path.exists('./logs'):
    os.makedirs('./logs')
    if not os.path.exists('./logs/ood'):
        os.makedirs('./logs/ood')
if not os.path.exists('./logs/ood/'+ args.dataset):
    os.makedirs('./logs/ood/'+ args.dataset)

stats_log = open('./logs/ood/' + args.dataset + '/ood_log' + '.txt', 'w')

use_gpu = torch.cuda.is_available()
options.update(
    {
        'use_gpu': use_gpu
    }
)

out_dataset = datasets.create(options['out_dataset'], **options)
dataset = datasets.create(options['dataset'], **options)
trainloader, testloader = dataset.trainloader, dataset.testloader
outloader = out_dataset.testloader
options.update(
    {
        'num_classes': dataset.num_classes
    }
)

if options['ood_dataset'] is None:
    if options['dataset'] == 'cifar10':
        options['ood_dataset'] = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100']#, 'interp']
        options['image_size'] = (32, 32, 3)
ood_eval = True
ood_test_loader = dict()
kwargs = {'pin_memory': False, 'num_workers': 40}
for ood in options['ood_dataset']:
    ood_test_set = get_dataset(options, dataset=ood, test_only=True, image_size=options['image_size'], eval=ood_eval)
    ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=options['batch_size'], **kwargs)

model = ResNet18(options['num_classes'], 16).cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark=True

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

bestauc = 0
iter = 0
aucs = []
accs = []
start_time = time.time()
for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 50:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    iter = train(epoch, model, optimizer, trainloader, iter)
    if epoch % 10 == 0:
        aucs = []
        accs = []
        for d in options['ood_dataset']:
            acc1, auc1 = test(model, d)

            aucs.append(auc1)
            accs.append(acc1)

            print("E[%d] [%s] AUC1 Err: [%.3f] ACC1 : [%.2f]\n" % (epoch, d, auc1, acc1 * 10))
            if epoch % 20 == 0:
                stats_log.write("------------Epoch %d-----------" % epoch)
                stats_log.write("E[%d] [%s] AUC1 Err : [%.3f] ACC1 : [%.2f]\n" % (epoch, d, auc1, acc1 * 10))
                stats_log.write("-------------------------")
        auc_results.append(aucs)
        acc_results.append(accs)
end_time = time.time()
print('time elapsed:', end_time - start_time)