# Out-of-Distribution-Baseline

## This is the repository for training and evaluating Out-of-Distribution Detection.

For simplicity, we implement the situation where CIFAR-10 dataset is used as in-distribution dataset and SVHN, CIFAR100, LSUN, ImageNet as out-distribution datasets.

Users can simply refer to datasets/datasets and datasets/ood_datasets to further apply their training scheme to combinations of other datasets.

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.2 ~ 1.7.1 
- [Torchvision] 0.4.0 ~ 0.8.2 depending on the version of torch.
- [scikit-learn](https://scikit-learn.org/stable/)

Other torch versions might work but we have not tested.


### Training 

We provide training example with this repo:


```bash
python ood_baseline.py
```

Different parameters, e.g. Epoch, BatchSize, and etc, can be adjusted with the arguments.
Check arguments at the top of ood_baseline.py

