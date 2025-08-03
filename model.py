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

# 设置随机种子保证可复现性
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
dgl.random.seed(SEED)
# 工具函数
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class EnhancedGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=4,dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        #输入层
        self.layers.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.dropout1 = nn.Dropout(dropout)
        #中间层
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.add_module(f"dropout{_ + 2}", nn.Dropout(dropout))
        #输出层
        self.layers.append(GraphConv(hid_dim, out_dim, norm='both'))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, graph, x):
        for i, conv in enumerate(self.layers[:-1]):
            x = F.gelu(conv(graph, x))
            if i < len(self.layers) - 2:  # 不在最后一层前应用Dropout
                x = getattr(self, f"dropout{i + 1}")(x)
        x = self.layers[-1](graph, x)
        return self.norm(x)

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

class DGN(nn.Module):
    def __init__(self, in_dim=230, hid_dim=512, out_dim=384,
                 num_classes=2, temp=0.5, num_layers=3,dropout=0.2):
        super().__init__()
        self.feature_noise = nn.Dropout(0.1)
        # 双编码器架构
        self.encoder = EnhancedGCN(in_dim, hid_dim, out_dim, num_layers,dropout=dropout)
        self.encoder_t = copy.deepcopy(self.encoder)
        set_requires_grad(self.encoder_t, False)
        # 动量更新参数
        self.momentum = 0.999
        self.temp = temp
        self.attention = MultiHeadAttention(out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hid_dim // 2),
            nn.LayerNorm(hid_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim // 2, num_classes)
        )

        # 对称投影头
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.proj_t = copy.deepcopy(self.proj)
        set_requires_grad(self.proj_t, False)
        self.global_graph = None
        self.all_bag_indices = None

    def set_global_graph(self, graph, bag_indices):
        """设置全局图结构和包索引"""
        self.global_graph = graph
        self.all_bag_indices = bag_indices

    def to(self, device):
        """设备同步"""
        super().to(device)
        if self.global_graph is not None:
            self.global_graph = self.global_graph.to(device)
        if self.all_bag_indices is not None:
            self.all_bag_indices = [ind.to(device) for ind in self.all_bag_indices]
        return self

    def forward(self, feat, bag_indices, labels=None):
        feat=self.feature_noise(feat)
        # 动量更新目标网络
        self._momentum_update()

        # 编码特征
        h = self.encoder(self.global_graph, feat)
        with torch.no_grad():
            h_t = self.encoder_t(self.global_graph, feat)
        # 对比学习损失
        z = F.normalize(self.proj(h), dim=-1)
        z_t = F.normalize(self.proj_t(h_t), dim=-1)
        contrast_loss = self.contrastive_loss(z, z_t)
        # 注意力聚合
        weights = F.softmax(self.attention(h), dim=0)
        bag_feats = [torch.sum(weights[nodes] * h[nodes], 0) for nodes in bag_indices]
        # 分类预测
        logits = self.classifier(torch.stack(bag_feats))
        # 总损失
        total_loss = contrast_loss * 0.6
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels, label_smoothing=0.1) * 0.4
            total_loss += cls_loss
        return logits, total_loss
    def contrastive_loss(self, q, k):
        logits = torch.mm(q, k.t()) / self.temp
        labels = torch.arange(len(q), device=q.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    def _momentum_update(self):
        """动量更新目标网络"""
        for param, target_param in zip(self.encoder.parameters(), self.encoder_t.parameters()):
            target_param.data = target_param.data * self.momentum + param.data * (1 - self.momentum)
        for param, target_param in zip(self.proj.parameters(), self.proj_t.parameters()):
            target_param.data = target_param.data * self.momentum + param.data * (1 - self.momentum)

def train_high_acc_model(all_features, global_graph, bag_indices, labels,
                         device='cuda', num_epochs=100):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    results = {'acc': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(bag_indices, labels)):
        print(f"\n=== Fold {fold + 1}/10 ===")

        # 初始化
        model = DGN().to(device)
        model.set_global_graph(global_graph.to(device), bag_indices)
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=0.05)

        # 改进方案（余弦退火 + 热重启）：
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # 每20轮重启一次
            T_mult=2  # 周期倍增
        )

        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler()

        # 早停机制
        best_acc = 0.0
        patience = 0

        for epoch in range(num_epochs):
            # 训练阶段
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
            scheduler.step()

            # 验证阶段
            model.eval()
            with torch.no_grad():
                logits, _ = model(all_features, [bag_indices[i] for i in val_idx])
                preds = logits.argmax(dim=1).cpu()
                acc = accuracy_score(labels[val_idx], preds)

                if acc > best_acc + 0.005:
                    best_acc = acc
                    patience = 0
                    torch.save(model.state_dict(), f"best_fold{fold + 1}.pth")
                else:
                    patience += 1
            # 进度打印
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:03d} | Loss: {loss.item():.4f} | "
                      f"Val Acc: {acc:.2%} | Best: {best_acc:.2%}")
        model.load_state_dict(torch.load(f"best_fold{fold + 1}.pth"))
        with torch.no_grad():
            logits, _ = model(all_features, [bag_indices[i] for i in val_idx])
            final_acc = accuracy_score(labels[val_idx], logits.argmax(dim=1).cpu())
            results['acc'].append(final_acc)
    print("\n=== Final Results ===")
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