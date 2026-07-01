import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# -------------------- 模型定义 --------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.encoder(x)

class WNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, q, y):
        return self.encoder(torch.cat([q, y], dim=1))

class CascadedPINN(nn.Module):
    def __init__(self, config, scalers):
        super().__init__()
        # 物理参数（不可训练）
        self.EI = nn.Parameter(torch.tensor(config["EI"]), requires_grad=False)
        self.KGA = nn.Parameter(torch.tensor(config["KGA"]), requires_grad=False)
        self.lambda_pde = nn.Parameter(torch.tensor(config["lambda_pde"]), requires_grad=False)
        self.sig_q = nn.Parameter(torch.tensor(config["sig_q"], dtype=torch.float32), requires_grad=False)
        self.mu_q = nn.Parameter(torch.tensor(config["mu_q"], dtype=torch.float32), requires_grad=False)

        # 从scalers中获取归一化参数
        self.sig_y = nn.Parameter(torch.tensor(scalers['X']['scale'][-1], dtype=torch.float32), requires_grad=False)
        self.mu_y = nn.Parameter(torch.tensor(scalers['X']['mean'][-1], dtype=torch.float32), requires_grad=False)
        self.sig_w = nn.Parameter(torch.tensor(scalers['w']['scale'], dtype=torch.float32), requires_grad=False)
        self.mu_w = nn.Parameter(torch.tensor(scalers['w']['mean'], dtype=torch.float32), requires_grad=False)

        self.q_net = QNetwork()
        self.w_net = WNetwork()

    def forward(self, x):
        y = x[:, -1:]
        q = self.q_net(x)
        w = self.w_net(q, y)
        return q, w

# -------------------- 数据预处理 --------------------
def load_and_preprocess_for_prediction(data_path, w_scale=0.001):
    data = pd.read_excel(data_path)
    X_features = data.iloc[:, 1:16]          # 前15个推进/地层参数
    y = data.iloc[:, -1].values.reshape(-1, 1)   # 位置（最后一列）
    displacement = data.iloc[:, -2].values.reshape(-1, 1) * w_scale   # 沉降 (mm→m)

    X = pd.concat([X_features, pd.DataFrame(y, columns=['y'])], axis=1)

    # 标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_w = StandardScaler()
    w_scaled = scaler_w.fit_transform(displacement)

    scalers = {
        'X': {'scale': scaler_X.scale_, 'mean': scaler_X.mean_},
        'w': {'scale': scaler_w.scale_[0], 'mean': scaler_w.mean_[0]}
    }
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(w_scaled, dtype=torch.float32), scalers, data

# -------------------- 主预测函数 --------------------
def predict_from_model(weight_path, data_path, param_path):
    # 1. 读取模型参数（EI, KGA等）
    param_df = pd.read_excel(param_path)
    config = {
        "EI": param_df['EI'].iloc[0],
        "KGA": param_df['KGA'].iloc[0],
        "lambda_pde": param_df['lambda_pde'].iloc[0],
        "sig_q": param_df['sig_q'].iloc[0],
        "mu_q": param_df['mu_q'].iloc[0]
    }

    # 2. 加载数据并预处理，获取标准化后的数据及统计量
    X_scaled, w_scaled, scalers, raw_data = load_and_preprocess_for_prediction(data_path)

    # 3. 创建模型并加载权重
    model = CascadedPINN(config, scalers)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    # 4. 预测
    with torch.no_grad():
        q_pred, w_pred_norm = model(X_scaled)
        # 反归一化得到实际沉降（米）
        w_pred = w_pred_norm * scalers['w']['scale'] + scalers['w']['mean']

    # 5. 提取真实沉降（原始米制）
    w_true = raw_data.iloc[:, -2].values * 0.001  # 原为毫米，转米

    # 6. 输出结果（保留环号）
    results = pd.DataFrame({
        '环号': raw_data.iloc[:, 0].values,
        '真实沉降(m)': w_true * 1000,
        '预测沉降(m)': w_pred.numpy().flatten() * 1000,
        '误差(mm)': (w_pred.numpy().flatten() - w_true) * 1,
        '真实沉降(反归一化前)': w_scaled.numpy().flatten(),  # 新增
        '预测沉降(反归一化前)': w_pred_norm.numpy().flatten()  # 新增
    })
    #print(w_true[96:121]*1000)
    #print(w_pred[96:121].numpy().flatten() * 1000)

    # 7. 计算R²
    ss_res = np.sum((results['真实沉降(m)'] - results['预测沉降(m)']) ** 2)
    ss_tot = np.sum((results['真实沉降(m)'] - np.mean(results['真实沉降(m)'])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    print(f"全局 R² = {r2:.4f}")

    ss_res2 = np.sum((w_true[96:121]*1000 - w_pred[96:121].numpy().flatten() * 1000) ** 2)
    ss_tot2 = np.sum((w_true[96:121]*1000 - np.mean(w_pred[96:121].numpy().flatten() * 1000)) ** 2)
    r22 = 1 - ss_res2 / ss_tot2 if ss_tot != 0 else 0.0
    print(f"测试集 R² = {r22:.4f}")

    return results

# -------------------- 使用 --------------------
if __name__ == "__main__":
    # 请根据实际路径修改
    weight_file = "model_weights.pth"
    data_file = "all.xls"
    param_file = "model_params.xlsx"

    if not os.path.exists(weight_file):
        print(f"权重文件不存在：{weight_file}")
    else:
        pred_results = predict_from_model(weight_file, data_file, param_file)
        pred_results.to_excel("prediction_results.xlsx", index=False)
        print("预测结果已保存到 prediction_results.xlsx")
        #print(pred_results.head())