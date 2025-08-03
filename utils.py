import pandas as pd
import torch
import numpy as np
import dgl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def load_musk_data(file_path):
    # ================== 数据读取 ==================
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            molecule = parts[0].strip('"')
            features = list(map(float, parts[2:-1]))
            label = int(parts[-1].rstrip('.'))
            data.append({
                'molecule': molecule,
                'features': features,
                'label': label
            })
    # ================== 数据标准化 ==================
    df = pd.DataFrame(data)
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(df['features'].tolist())
    all_features = torch.tensor(standardized_features, dtype=torch.float32)
    # ================== 自适应图构建 ==================
    def build_reciprocal_knn(features, k=10):
        sim_matrix = cosine_similarity(features.numpy())
        n = sim_matrix.shape[0]
        adj = np.zeros((n, n))
        for i in range(n):
            # 获取top k相似节点
            top_k = np.argpartition(sim_matrix[i], -k)[-k:]
            for j in top_k:
                # 检查是否互为邻居
                if i in np.argpartition(sim_matrix[j], -k)[-k:]:
                    adj[i][j] = 1
        return dgl.from_scipy(sparse.csr_matrix(adj))
    n_samples = all_features.shape[0]
    base_k = max(5, int(np.log2(n_samples)))  # 对数缩放基础k值
    global_graph = build_reciprocal_knn(all_features, k=min(base_k, 15))
    # ================== 包结构生成 ==================
    labels = []
    bag_indices = []
    # 按分子名分组
    grouped = df.groupby('molecule')
    for name, group in grouped:
        # 标签校验（同一包内标签应一致）
        assert group['label'].nunique() == 1, f"包 {name} 存在矛盾标签"
        labels.append(group['label'].iloc[0])
        # 获取索引并校验
        indices = group.index.tolist()
        assert all(df.loc[indices, 'molecule'] == name), "索引与分子名不匹配"
        bag_indices.append(torch.LongTensor(indices))
    return all_features, global_graph, bag_indices, torch.LongTensor(labels)