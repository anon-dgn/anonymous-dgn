'''去掉对比学习和动量更新'''
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

class BaselineModel(nn.Module):
    def __init__(self, in_dim=230, hid_dim=512, out_dim=384,
                 num_classes=2, dropout=0.2):
        super().__init__()
        # 仅保留单编码器
        self.encoder = EnhancedGCN(in_dim, hid_dim, out_dim, dropout=dropout)
        self.feature_noise = nn.Dropout(0.1)
        # 简化注意力机制
        self.attention = MultiHeadAttention(out_dim)
        # 分类器保持不变
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hid_dim // 2),
            nn.LayerNorm(hid_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim // 2, num_classes)
        )
        # 移除对比学习相关组件
        self.global_graph = None
        self.all_bag_indices = None
    def set_global_graph(self, graph, bag_indices):
        """设置全局图结构和包索引"""
        self.global_graph = graph
        self.all_bag_indices = bag_indices

    def to(self, device):
        """确保全局图与模型在同一设备"""
        super().to(device)
        if self.global_graph is not None:
            self.global_graph = self.global_graph.to(device)
        if self.all_bag_indices is not None:
            self.all_bag_indices = [ind.to(device) for ind in self.all_bag_indices]
        return self

    def forward(self, feat, bag_indices, labels=None):
        feat = self.feature_noise(feat)
        # 单编码器前向
        h = self.encoder(self.global_graph, feat)
        # 注意力聚合
        weights = F.softmax(self.attention(h), dim=0)
        bag_feats = [torch.sum(weights[nodes] * h[nodes], 0) for nodes in bag_indices]
        # 分类预测
        logits = self.classifier(torch.stack(bag_feats))
        if labels is not None:
            total_loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=0.1
            )
        else:
            total_loss = None
        return logits, total_loss


def train_baseline_model(all_features, global_graph, bag_indices, labels,
                         device='cuda', num_epochs=150):
    # 保持与原始实验一致的随机种子
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dgl.random.seed(SEED)

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    results = {'acc': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(bag_indices, labels)):
        print(f"\n=== Fold {fold + 1}/10 ===")
        # 初始化Baseline模型
        model = BaselineModel().to(device)
        model.set_global_graph(global_graph.to(device), bag_indices)
        # 统一优化器配置
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.05
        )
        # 学习率调度器
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2
        )
        # 早停机制参数
        best_acc = 0.0
        patience = 0
        dynamic_patience = max(15, int(0.2 * num_epochs))
        # 训练循环
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            # 前向计算
            train_bags = [bag_indices[i] for i in train_idx]
            logits, loss = model(
                all_features,
                train_bags,
                labels=labels[train_idx].to(device)
            )
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_bags = [bag_indices[i] for i in val_idx]
                logits, _ = model(all_features, val_bags)
                preds = logits.argmax(dim=1).cpu()
                acc = accuracy_score(labels[val_idx], preds)
                # 早停判断
                if acc > best_acc + 0.005:
                    best_acc = acc
                    patience = 0
                    torch.save(model.state_dict(), f"baseline_best_fold{fold + 1}.pth")
                else:
                    patience += 1
            # 训练进度打印
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:03d} | Loss: {loss.item():.4f} | "
                      f"Val Acc: {acc:.2%} | Best: {best_acc:.2%}")
        # 最终评估
        model.load_state_dict(torch.load(f"baseline_best_fold{fold + 1}.pth"))
        with torch.no_grad():
            val_bags = [bag_indices[i] for i in val_idx]
            logits, _ = model(all_features, val_bags)
            final_acc = accuracy_score(labels[val_idx], logits.argmax(dim=1).cpu())
            results['acc'].append(final_acc)
    # 结果报告
    print("\n=== Baseline Results ===")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    graph = graph.to(device)
    results = train_baseline_model(
        all_features=features,
        global_graph=graph,
        bag_indices=bags,
        labels=labels,
        num_epochs=150
    )