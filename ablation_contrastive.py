'''消融实验 消除对比学习部分'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import dgl
import numpy as np
from utils import load_musk_data
from dgl.nn import GraphConv
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
dgl.random.seed(SEED)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
# 保持GCN编码器不变
class EnhancedGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=4, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.dropout1 = nn.Dropout(dropout)
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.add_module(f"dropout{_ + 2}", nn.Dropout(dropout))
        self.layers.append(GraphConv(hid_dim, out_dim, norm='both'))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, graph, x):
        for i, conv in enumerate(self.layers[:-1]):
            x = F.gelu(conv(graph, x))
            if i < len(self.layers) - 2:
                x = getattr(self, f"dropout{i + 1}")(x)
        x = self.layers[-1](graph, x)
        return self.norm(x)
# 保持多头注意力不变
class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        return sum(head(x) for head in self.heads) / len(self.heads)
# 移除对比学习的主模型
class HighAccMILModel(nn.Module):
    def __init__(self, in_dim=230, hid_dim=512, out_dim=384,
                 num_classes=2, num_layers=3, dropout=0.2):
        super().__init__()
        self.feature_noise = nn.Dropout(0.1)
        # 保持双编码器架构
        self.encoder = EnhancedGCN(in_dim, hid_dim, out_dim, num_layers, dropout=dropout)
        self.encoder_t = copy.deepcopy(self.encoder)
        set_requires_grad(self.encoder_t, False)
        # 移除投影头和相关参数
        self.momentum = 0.999
        self.attention = MultiHeadAttention(out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hid_dim // 2),
            nn.LayerNorm(hid_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim // 2, num_classes)
        )
        self.global_graph = None
        self.all_bag_indices = None
    def set_global_graph(self, graph, bag_indices):
        self.global_graph = graph
        self.all_bag_indices = bag_indices
    def to(self, device):
        super().to(device)
        if self.global_graph is not None:
            self.global_graph = self.global_graph.to(device)
        if self.all_bag_indices is not None:
            self.all_bag_indices = [ind.to(device) for ind in self.all_bag_indices]
        return self
    def forward(self, feat, bag_indices, labels=None):
        feat = self.feature_noise(feat)
        # 保持动量更新
        self._momentum_update()
        # 编码特征
        h = self.encoder(self.global_graph, feat)
        with torch.no_grad():
            h_t = self.encoder_t(self.global_graph, feat)
        weights = F.softmax(self.attention(h), dim=0)
        bag_feats = [torch.sum(weights[nodes] * h[nodes], 0) for nodes in bag_indices]
        # 分类预测
        logits = self.classifier(torch.stack(bag_feats))
        # 仅保留分类损失
        total_loss = F.cross_entropy(logits, labels, label_smoothing=0.1) if labels is not None else 0
        return logits, total_loss
    def _momentum_update(self):
        """仅更新encoder_t参数"""
        for param, target_param in zip(self.encoder.parameters(), self.encoder_t.parameters()):
            target_param.data = target_param.data * self.momentum + param.data * (1 - self.momentum)

def train_high_acc_model(all_features, global_graph, bag_indices, labels,
                         device='cuda', num_epochs=150):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    results = {'acc': []}
    for fold, (train_idx, val_idx) in enumerate(skf.split(bag_indices, labels)):
        print(f"\n=== Fold {fold + 1}/10 ===")

        model = HighAccMILModel().to(device)
        model.set_global_graph(global_graph.to(device), bag_indices)

        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=0.05)
        scaler = torch.cuda.amp.GradScaler()
        best_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                _, loss = model(all_features,
                                [bag_indices[i] for i in train_idx],
                                labels[train_idx].to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            scaler.step(optimizer)
            scaler.update()
            # 验证阶段
            model.eval()
            with torch.no_grad():
                logits, _ = model(all_features, [bag_indices[i] for i in val_idx])
                acc = accuracy_score(labels[val_idx], logits.argmax(dim=1).cpu())

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f"best_fold{fold + 1}.pth")

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.2%}")

        # 最终评估
        model.load_state_dict(torch.load(f"best_fold{fold + 1}.pth"))
        with torch.no_grad():
            logits, _ = model(all_features, [bag_indices[i] for i in val_idx])
            final_acc = accuracy_score(labels[val_idx], logits.argmax(dim=1).cpu())
            results['acc'].append(final_acc)

    print(f"Average Accuracy: {np.mean(results['acc']):.2%} ± {np.std(results['acc']):.2%}")
    return results
if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    dgl.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 加载数据
    features, graph, bags, labels = load_musk_data("datasets/FOX.data")
    # 数据预处理
    features = torch.FloatTensor(StandardScaler().fit_transform(features))
    labels = torch.LongTensor(labels)
    # 转换到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    graph = graph.to(device)
    # 训练
    results = train_high_acc_model(
        all_features=features,
        global_graph=graph,
        bag_indices=bags,
        labels=labels,
        num_epochs=100
    )