import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
import numpy as np
import copy
from torch.distributions.normal import Normal
import os
import math
import glob
import warnings
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss

try:
    from peft import get_peft_config, get_peft_model
except:
    print('install peft if you use LoRA')

import source
from source.model import get_model
from source.load_checkpoint import load_checkpoint
# from source.utils import setup_seed
import rasterio


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


class_rgb = {
    "bg": [0, 0, 0],
    "tree": [34, 97, 38],
    "rangeland": [0, 255, 36],
    "bareland": [128, 0, 0],
    "agric land type 1": [75, 181, 73],
    "road type 1": [255, 255, 255],
    "sea, lake, & pond": [0, 69, 255],
    "building type 1": [222, 31, 7],
}

class_gray = {
    "bg": 0,
    "tree": 1,
    "rangeland": 2,
    "bareland": 3,
    "agric land type 1": 4,
    "road type 1": 5,
    "sea, lake, & pond": 6,
    "building type 1": 7,
}


def label2rgb(a):
    """
    a: labels (HxW)
    """
    out = np.zeros(shape=a.shape + (3, ), dtype="uint8")
    for k, v in class_gray.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]
    return out


class EdgeValuePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgeValuePredictor, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.edge_predictor_mean = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        self.edge_predictor_logvar = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        combined = torch.cat([src, dst], dim=1)
        mean = self.edge_predictor_mean(combined)
        logvar = self.edge_predictor_logvar(combined)
        var = torch.exp(0.5 * logvar)
        return mean, var


def grpo_loss(advantages, logits, old_logits, beta):
    """
    rewards: 采样得到的奖励
    advantages: 优势估计
    logits: 新策略的 logits
    old_logits: 旧策略的 logits
    beta: KL 散度的权重系数
    """
    # 计算概率比
    # 去掉大小为1 的维度
    logits = logits.squeeze()
    old_logits=old_logits.squeeze()
    logratio = logits - old_logits
    ratio = torch.exp(logratio)
    # 计算剪裁后的概率比
    clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)

    # 计算策略的最小化目标
    policy_loss = torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

    # 计算 KL 散度正则化项
    kl_divergence = ((1/ratio - 1) + logratio).mean()
    kl_loss = beta * kl_divergence

    # 总损失
    total_loss = -(policy_loss - kl_loss)
    return total_loss

metric = source.metrics.IoU2()

def cal_reward(network, edge_weight, edge_index, graph_labels, feature, mean_features, val_pth):
    with torch.no_grad():
        feature_ori = feature
        msk_gt = load_grayscale(val_pth['label'][0])
        h,w=msk_gt.shape
        mean_features=np.array(mean_features)
        weights=np.zeros(mean_features.shape[0])  #num_nodes, num_features
        weighted_features=np.zeros_like(mean_features)  #num_nodes, num_features

        for i_node in range(len(edge_index[0])):
            node=edge_index[0][i_node]
            weighted_features[node]+= mean_features[node]*edge_weight[i_node].numpy()
            weights[node]+=edge_weight[i_node]

        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                node=graph_labels[i][j]
                feature[i][j] = 0.1 * weighted_features[node]+ (1-0.1*weights[node])*feature[i][j]
        feature = torch.tensor(feature).to('cuda').unsqueeze(0)
        msk_gt = torch.tensor(msk_gt).to('cuda').unsqueeze(0)
        
        msk_pred = network.classifier(feature)  
        # 插值到相同大小
        msk_pred = F.interpolate(msk_pred, size=(h, w), mode='bilinear', align_corners=False)
        reward = metric(msk_pred, msk_gt)

    return reward


def train_grpo(model,model_old, data, optimizer, network, graph_labels, feature, mean_features, val_pth, beta=0.01, num_samples=16):
    model.train()
    optimizer.zero_grad()
    
    with torch.no_grad():
        old_mean, old_var = model_old(data.x, data.edge_index)
        
        
    rewards =[]
    logProb_olds = []
    logProbs = []
    mean, var = model(data.x, data.edge_index)
    mean = mean/10
    old_mean = old_mean/10
    
    for _ in range(num_samples):
        eps = torch.randn_like(mean)
        eps = torch.randn_like(mean)
        sampled_value = mean + eps * var
        logProb_old = Normal(old_mean, old_var).log_prob(sampled_value).detach()
        logProb = Normal(mean, var).log_prob(sampled_value)
        logProb_olds.append(logProb_old)
        logProbs.append(logProb)
        # 计算奖励
        edge_weight = torch.sigmoid(sampled_value).detach()   # (num_edge, 1)
        reward = cal_reward(network, edge_weight, data.edge_index, graph_labels, feature, mean_features, val_pth)  # 奖励是实际权重和预测权重的差距
        # 使用组内平均奖励作为基准
        rewards.append(reward.item())

    rewards = torch.tensor(rewards, dtype=torch.float32)
    logProbs = torch.stack(logProbs).mean(dim=(-1,-2))
    logProb_olds = torch.stack(logProb_olds).mean(dim=(-1,-2))
    # 计算优势
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)/num_samples  # 标准化奖励
    # 计算损失
    loss = grpo_loss(advantages, logProbs, logProb_olds, beta)
    loss.backward()
    optimizer.step()
    
    return loss.item(), rewards.mean()

def test_grpo(model, data, edge_index):
        # 测试模型
    model.eval()
    with torch.no_grad():
        mean, var = model(data.x, data.edge_index)
        eps = torch.randn_like(mean)
        sampled_value = mean
        edge_weight = torch.sigmoid(sampled_value)
    #计算所有边预测结果的准确率
    correct = ((edge_weight > 0.5) == (data.edge_attr > 0.5)).float().sum()
    print(f"预测准确率: {correct.item() / len(data.edge_attr):.4f}")
    # print("\n边预测结果:")
    # for i in range(len(data.edge_attr)):
    #     print(f"边 {edge_index[:, i]}: 实际权重={data.edge_attr[i].item():.4f}, 预测权重={edge_weight[i].item():.4f}")




def get_pred_data(network, n_classes):
    
    # 生成图数据
    val_pth={
        'img': [],
        'label': [],
        'pred': [],
    }
    with open('/cephfs/shared/ruixun/project/Test/Image_segmentation/data/stage1_val.txt', 'r') as f:
        for line in f.readlines():
            val_pth['img'].append('/cephfs/shared/ruixun/project/Test/Image_segmentation/data/trainset/images/'+line.strip())
            val_pth['label'].append('/cephfs/shared/ruixun/project/Test/Image_segmentation/data/trainset/labels/'+line.strip())
            
    for fn_img in val_pth['img']:
        img = source.dataset.load_multiband(fn_img)
        h, w = img.shape[:2]
        power = math.ceil(np.log2(h) / np.log2(2))
        shape = (2**power, 2**power)
        img = cv2.resize(img, shape)
        input = TF.to_tensor(img).unsqueeze(0).float().to('cuda')

        pred = []
        with torch.no_grad():
            msk = network(input)
            pred = msk.squeeze().cpu().numpy()
        pred = pred.argmax(axis=0).astype("uint8")
        size = pred.shape[0:]
        y_pr = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        #保存y_pr为npy文件
        OUT_DIR='/cephfs/shared/ruixun/project/Test/Image_segmentation/result/UPerNet-tu-convnext_large_mlp.clip_laion2b_ft_soup_320_s0_CE_DICE'
        np.save(os.path.join(OUT_DIR, os.path.basename(fn_img) + '.npy'), y_pr)
        val_pth['pred'].append(os.path.join(OUT_DIR, os.path.basename(fn_img) + '.npy'))
        
        # save image as png
        filename = os.path.splitext(os.path.basename(fn_img))[0]
        y_pr_rgb = label2rgb(y_pr)
        Image.fromarray(y_pr_rgb).save(os.path.join(OUT_DIR, filename + '.png'))
    return val_pth



def get_Graph_data(network, val_pth):
    # 读取图数据
    for i in range(len(val_pth['img'])):
        img = source.dataset.load_multiband(val_pth['img'][i])
        
        h, w = img.shape[:2]
        power = math.ceil(np.log2(h) / np.log2(2))
        shape = (2**power, 2**power)
        img = cv2.resize(img, shape)
        input = TF.to_tensor(img).unsqueeze(0).float().to('cuda')

        with torch.no_grad():
            feature = network.model(input)
            msk = network.classifier(feature)
            msk = msk.squeeze().cpu().numpy()
        pred = msk.argmax(axis=0).astype("uint8")
        _, binary_image = cv2.threshold(pred, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary_image)
        # 计算每个连通区域的msk取值的均值
        mean_values = []
        mean_features = []
        feature = feature.squeeze().cpu().numpy()
        for j in range(num_labels):
            mean_value = np.mean(msk[:,labels == j],axis=1)
            mean_feature = np.mean(feature[:,labels == j],axis=1)
            mean_values.append(mean_value)
            mean_features.append(mean_feature)
        
        # 建立节点之间的连接关系，基于连通域相邻关系
        adjacency_list = [[] for _ in range(num_labels)]  # 邻接表表示图结构
        # 遍历每个像素来确定连通域的相邻关系
        height, width = labels.shape
        for i in range(height):
            for j in range(width):
                current_label = labels[i, j]
                if current_label == 0:  # 跳过背景（假设标签 0 代表背景）
                    continue
                # 检查上下左右四个方向的相邻像素
                for dx, dy in [(-5, 0), (5, 0), (0, -5), (0, 5), (-5, -5), (5, 5), (5, -5), (-5, 5)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < height and 0 <= ny < width:
                        neighbor_label = labels[nx, ny]
                        if neighbor_label != 0 and neighbor_label != current_label:
                            # 如果邻居连通域不在当前连通域的邻接列表中，则添加
                            if neighbor_label not in adjacency_list[current_label]:
                                adjacency_list[current_label].append(neighbor_label)

        # 输出图结构
        for idx, neighbors in enumerate(adjacency_list):
            print(f"节点 {idx} 与节点 {neighbors} 相邻")
        # 将图结构转换为边索引
        edge_index = []
        for idx, neighbors in enumerate(adjacency_list):
            for neighbor in neighbors:
                edge_index.append([idx, neighbor])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # 图节点特征
        x = torch.tensor(mean_values, dtype=torch.float)
        break

    return x, labels, edge_index, feature, mean_features
    


def main():
    # 创建一个简单的图数据
    # num_nodes = 12
    num_features = 8
    # edge_prob = 0.3
    # x = torch.randn(num_nodes, num_features)
    # adj_matrix = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[1 - edge_prob, edge_prob])
    # adj_matrix = np.triu(adj_matrix)
    # adj_matrix = adj_matrix + adj_matrix.T
    # adj_matrix = torch.from_numpy(adj_matrix).to(torch.int64)
    # edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
    

    # seed = 0
    # setup_seed(seed)
    classes = [0, 1, 2, 3, 4, 5, 6, 7]
    n_classes = len(classes)
    # torch.manual_seed(42)
    network = get_model(
    dict(model_name='UPerNet',
         encoder_name='tu-convnext_large_mlp.clip_laion2b_ft_soup_320',
         encoder_weights=None,
         encoder_depth=4,
         classes=n_classes))
    checkpoint_path = '/cephfs/shared/ruixun/project/Test/pretrained/open_clip_pytorch_model.bin'
    network = load_checkpoint('convnext-clip',
                            network,
                            checkpoint_path,
                            strict=True)
    network.load_state_dict(torch.load('/cephfs/shared/ruixun/project/Test/Image_segmentation/weight/UPerNet-tu-convnext_large_mlp.clip_laion2b_ft_soup_320_s0_CE_DICE.pth'), strict=False)
    network.eval()
    network.to('cuda')
    for param in network.parameters():
        param.requires_grad = False
    val_pth = get_pred_data(network, n_classes)
    x, graph_labels, edge_index, feature, mean_features = get_Graph_data(network, val_pth)
    edge_values = torch.randn(len(edge_index[0]))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_values)

    # 初始化模型和优化器
    model = EdgeValuePredictor(in_channels=num_features, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    print("开始训练模型...")
    model_old = copy.deepcopy(model)
    for epoch in range(50):
        torch.manual_seed(epoch)
        loss, rewards= train_grpo(model,model_old, data, optimizer, network, graph_labels, feature, mean_features, val_pth)
        print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, reward_mean: {rewards.mean():.4f}')
        
        

if __name__ == '__main__':
    main()