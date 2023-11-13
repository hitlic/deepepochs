'''
利用图神经网络对节点进行分类
需安装DGL：
    conda install -c dglteam dgl
    或者
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
'''
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import dgl.data
from dgl.nn import GraphConv
from torch.utils.data import DataLoader
from deepepochs import Trainer

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def loss(preds, targets):
    labels, masks = targets
    return F.cross_entropy(preds[masks], labels[masks])


def acc(preds, targets):
    labels, masks = targets
    return (preds.argmax(1)[masks] == labels[masks]).float().mean()


def collate_fn(batch, mask):
    g = batch[0]
    feats = g.ndata['feat']
    labels = g.ndata['label']
    masks = g.ndata[mask]
    return g, feats, (labels, masks)


dataset = dgl.data.CoraGraphDataset()
train_dl = DataLoader(dataset, batch_size=1, collate_fn=partial(collate_fn, mask='train_mask'))
val_dl =  DataLoader(dataset, batch_size=1, collate_fn=partial(collate_fn, mask='val_mask'))
test_dl =  DataLoader(dataset, batch_size=1, collate_fn=partial(collate_fn, mask='test_mask'))

feat_dim = dataset[0].ndata['feat'].shape[1]
model = GCN(feat_dim, 16, dataset.num_classes)

opt = torch.optim.Adam(model.parameters(), lr=0.01)

trainer = Trainer(model, loss, opt, 100, metrics=[acc])
trainer.fit(train_dl, val_dl)
trainer.test(test_dl)
