import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas import  ExcelWriter
import os
import random
from skopt import gp_minimize # Gaussian Process based minimization
from skopt.space import Real, Integer # For defining parameter ranges
from skopt.utils import use_named_args


# -------------------- 全局随机种子配置（新增代码） --------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 强制设置CUDA相关种子（即使不用GPU）
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- 模型定义 --------------------
class UnifiedNetwork(nn.Module):
    def __init__(self, input_dim=16): # 15 propulsion params + 1 y_coordinate
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh()
        )
        # Output head for q
        self.q_head = nn.Linear(16, 1)
        # Output head for w
        self.w_head = nn.Linear(16, 1)

    def forward(self, x):
        shared_features = self.shared_encoder(x)
        q_output = self.q_head(shared_features)
        w_output = self.w_head(shared_features)
        return q_output, w_output

class UnifiedPINN(nn.Module):
    def __init__(self, config, scalers):
        super().__init__()
        # 物理参数
        self.EI = nn.Parameter(torch.tensor(config["EI"]), requires_grad=False)
        self.KGA = nn.Parameter(torch.tensor(config["KGA"]), requires_grad=False)
        self.lambda_pde = nn.Parameter(torch.tensor(config["lambda_pde"]), requires_grad=False)
        # Renamed to avoid potential conflict with q output from network
        self.sig_q_param = nn.Parameter(torch.tensor(config["sig_q"], dtype=torch.float32), requires_grad=False)
        self.mu_q_param = nn.Parameter(torch.tensor(config["mu_q"], dtype=torch.float32), requires_grad=False)

        # 从 scalers['X'] 中获取y的归一化参数
        self.sig_y = nn.Parameter(
            torch.tensor(scalers['X']['scale'][-1], dtype=torch.float32),
            requires_grad=False
        )
        self.mu_y = nn.Parameter(
            torch.tensor(scalers['X']['mean'][-1], dtype=torch.float32),
            requires_grad=False
        )

        # 位移w的归一化参数
        self.sig_w = nn.Parameter(
            torch.tensor(scalers['w']['scale'], dtype=torch.float32),
            requires_grad=False
        )
        self.mu_w = nn.Parameter(
            torch.tensor(scalers['w']['mean'], dtype=torch.float32),
            requires_grad=False
        )

        # 网络结构
        self.unified_net = UnifiedNetwork()

    def forward(self, x):
        # Input x contains propulsion parameters and position y
        # The unified_net directly outputs q and w
        q, w = self.unified_net(x)
        return q, w

    def compute_q_derivatives(self, x, order=2):
        '''计算网络输出 q 对 y 的导数（支持二阶导数）'''
        q_output, _ = self.unified_net(x) # We only need q for this

        derivative = [q_output]
        current_derivation = q_output

        for _ in range(order):
            grad = autograd.grad(
                outputs=current_derivation,
                inputs=x,
                grad_outputs=torch.ones_like(current_derivation),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            grad_y = grad[:, -1:] # Extract gradient w.r.t. y (last column of x)
            derivative.append(grad_y)
            current_derivation = grad_y
        return tuple(derivative)

    def compute_w_derivatives(self, x, order=4):
        # 计算网络输出 w 对 y 的导数
        _, w_output = self.unified_net(x) # We only need w for this

        derivative = [w_output]
        current_derivation = w_output
        for _ in range(order):
            grad = autograd.grad(
                outputs=current_derivation,
                inputs=x,
                grad_outputs=torch.ones_like(current_derivation),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            grad_y = grad[:, -1:] # Extract gradient w.r.t. y (last column of x)
            derivative.append(grad_y)
            current_derivation = grad_y
        return tuple(derivative)

    def compute_loss(self, x, w_true):
        # 计算网络A的 q 及其导数
        q_pred, dq_dy, d2q_dy2 = self.compute_q_derivatives(x, order=2)

        # 计算网络B的 w 及其导数
        w_pred, dw_dy, d2w_dy2, d3w_dy3, d4w_dy4 = self.compute_w_derivatives(x, order=4)

        # PDE残差项系数计算
        a = self.sig_w / (self.sig_y ** 4)
        b = self.sig_q_param / (self.sig_y ** 2) * (1 / self.KGA) # Use renamed param
        c = self.sig_q_param / self.EI                            # Use renamed param
        d = self.mu_q_param / self.EI                             # Use renamed param
        c_term = c * q_pred + d

        # PDE残差公式
        pde_residual = (a * d4w_dy4 +
                        b * d2q_dy2 -
                        c_term)
        pde_loss = torch.mean(pde_residual ** 2)

        # 数据损失（同时考虑w和q的MSE）
        data_loss_w = torch.mean((w_true - w_pred) ** 2)

        # 总损失
        total_loss = data_loss_w + self.lambda_pde * pde_loss

        # 返回所有损失项
        return (q_pred.detach(),
                dq_dy.detach(),
                d2q_dy2.detach(),
                w_pred.detach(),
                dw_dy.detach(),
                d2w_dy2.detach(),
                d3w_dy3.detach(),
                d4w_dy4.detach(),
                a,
                b,
                c,
                d,
                pde_loss,
                data_loss_w,
                total_loss
                )

# -------------------- 数据预处理 --------------------
def load_and_preprocess(data_path, w_scale=0.001, train_ratio=0.826):
    """加载数据，合并推进参数和位置y作为输入特征，并进行归一化"""
    data = pd.read_excel(data_path)

    # 提取特征：假设前2-15列为推进参数，最后两列为位移w和位置y
    X_features = data.iloc[:, 1:16]  # 前8列推进参数
    y = data.iloc[:, -1].values.reshape(-1, 1)  # 最后一列为y
    displacement = data.iloc[:, -2].values.reshape(-1, 1) * w_scale  # 位移w转换为米

    # 合并推进参数和y作为输入X，共16维特征（15+1）
    X = pd.concat([X_features, pd.DataFrame(y, columns=['y'], index=X_features.index)], axis=1) # Ensure index alignment

    # -------------------- 归一化处理 --------------------
    scalers = {}
    # 对输入X进行归一化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scalers['X'] = {
        'scaler': scaler_X,
        'data': X_scaled,
        'scale': scaler_X.scale_,
        'mean': scaler_X.mean_
    }

    # 对位移w进行归一化
    scaler_w = StandardScaler()
    w_scaled = scaler_w.fit_transform(displacement)
    scalers['w'] = {
        'scaler': scaler_w,
        'data': w_scaled,
        'scale': scaler_w.scale_[0],
        'mean': scaler_w.mean_[0]
    }

    # -------------------- 数据分割 (Strictly as per original using slice) --------------------
    n_total = X_scaled.shape[0]
    n_train = int(n_total * train_ratio)
    train_indices = slice(0, n_train)
    val_indices = slice(n_train, None)

    # 转换为Tensor
    train_data = {
        'X_train': torch.tensor(X_scaled[train_indices], dtype=torch.float32, requires_grad=True),
        'w_train': torch.tensor(w_scaled[train_indices], dtype=torch.float32),
        'X_val': torch.tensor(X_scaled[val_indices], dtype=torch.float32, requires_grad=True),
        'w_val': torch.tensor(w_scaled[val_indices], dtype=torch.float32),
    }
    return train_data, scalers


# -------------------- 训练函数 （仅使用训练集）--------------------
def run_training(config, train_data, scalers, save_dir, num_epochs):
    # 初始化
    model = UnifiedPINN(config, scalers) # Use UnifiedPINN
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 创建结果容器
    loss_history = []
    pred_history = []

    # 获取数据分割
    X_train = train_data['X_train']
    w_train = train_data['w_train']

    X_val = train_data['X_val']
    w_val = train_data['w_val']

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 损失计算
        (q_pred,dq_dy,
         d2q_dy2,w_pred,
         dw_dy,d2w_dy2,
         d3w_dy3,d4w_dy4,
         a,b,c,d,
         pde_loss,data_loss_w,total_loss) = model.compute_loss(x=X_train,
                                     w_true=w_train)

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 记录当前epoch的损失
        epoch_data = {
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'data_loss_w': data_loss_w.item(),
            'pde_loss': config['lambda_pde'] * pde_loss.item(),
            'lambda_pde': config['lambda_pde'],
            'EI': config['EI'],
            'KGA': config['KGA'],
            "sig_q": config["sig_q"],
            "mu_q": config["mu_q"],
        }
        loss_history.append(epoch_data)

        # 在每100epoch的保存部分
        if epoch % 100 == 0:
            # === 训练集预测 ===
            train_pred_df = create_prediction_df(
                epoch=epoch,
                X=X_train,
                q_pred=q_pred, dq_dy=dq_dy, d2q_dy2=d2q_dy2,
                w_pred=w_pred, dw_dy=dw_dy, d2w_dy2=d2w_dy2,
                d3w_dy3=d3w_dy3, d4w_dy4=d4w_dy4,
                a=a, b=b, c=c, d=d,
                w_true=w_train,
                scalers=scalers,
                suffix='Train'
            )

            # === 验证集预测 ===
            model.eval()
            (q_val_pred, dq_dy_val, d2q_dy2_val,
             w_val_pred, dw_dy_val, d2w_dy2_val,
             d3w_dy3_val, d4w_dy4_val,
             a_val, b_val, c_val, d_val,
             pde_loss_val, data_loss_w_val,
             total_loss_val) = model.compute_loss( # total_loss_val is computed but not explicitly used here
                x=X_val,
                w_true=w_val
            )
            val_pred_df = create_prediction_df(
                epoch=epoch,
                X=X_val,
                q_pred=q_val_pred, dq_dy=dq_dy_val, d2q_dy2=d2q_dy2_val,
                w_pred=w_val_pred, dw_dy=dw_dy_val, d2w_dy2=d2w_dy2_val,
                d3w_dy3=d3w_dy3_val, d4w_dy4=d4w_dy4_val,
                a=a_val, b=b_val, c=c_val, d=d_val, w_true=w_val,
                scalers=scalers,
                suffix='Val'
            )
            # 合并预测结果
            combined_df = pd.concat([train_pred_df, val_pred_df])
            pred_history.append(combined_df)

    # 最终保存
    pd.DataFrame(loss_history).to_excel(
            os.path.join(save_dir, 'loss_history.xlsx'),
            index=False
            )

    if pred_history: # Ensure pred_history is not empty before writing
        with pd.ExcelWriter(os.path.join(save_dir, 'all_predictions.xlsx')) as writer:
            for i, df in enumerate(pred_history):
                df.to_excel(writer, sheet_name=f'epoch_{i * 100}', index=False)

    def compute_r2(y_true, y_pred):
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.detach().cpu().numpy().flatten() # Ensure detach for pred
        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    model.eval()
    with torch.no_grad():
        _, w_pred_train_final = model(train_data['X_train'])
        _, w_pred_val_final = model(train_data['X_val'])

    R2_train = compute_r2(train_data['w_train'], w_pred_train_final)
    R2_val = compute_r2(train_data['w_val'], w_pred_val_final)

    param_df = pd.DataFrame([{
        'EI': config['EI'],
        'KGA': config['KGA'],
        'lambda_pde': config['lambda_pde'],
        "sig_q": config["sig_q"],
        "mu_q": config["mu_q"],
        'sig_w': scalers['w']['scale'],
        'sig_y': scalers['X']['scale'][-1],
        'mu_w': scalers['w']['mean'],
        'mu_y': scalers['X']['mean'][-1],
        'R2_train': R2_train,
        'R2_val': R2_val
    }])
    param_df.to_excel(os.path.join(save_dir, 'model_params.xlsx'), index=False)

    return model


# -------------------- 预测数据保存  --------------------
def create_prediction_df(epoch, X,
                         q_pred, dq_dy, d2q_dy2,
                         w_pred, dw_dy, d2w_dy2,
                         d3w_dy3, d4w_dy4,
                         a, b, c, d,
                         w_true,
                         scalers, suffix):

    def squeeze_tensor(tensor):
        return tensor.detach().cpu().numpy().squeeze()

    return pd.DataFrame({
        'Dataset': [suffix] * len(X),
        'y': squeeze_tensor(X[:, -1].detach()), # Ensure X is detached here

        'pred_q': squeeze_tensor(q_pred),
        'dq/dy': squeeze_tensor(dq_dy),
        'd2q/dy2': squeeze_tensor(d2q_dy2),

        'pred_w': squeeze_tensor(w_pred),
        'true_w': squeeze_tensor(w_true),
        'dw/dy': squeeze_tensor(dw_dy),
        'd2w/dy2': squeeze_tensor(d2w_dy2),
        'd3w/dy3': squeeze_tensor(d3w_dy3),
        'd4w/dy4': squeeze_tensor(d4w_dy4),

        'term1': squeeze_tensor(a * d4w_dy4),
        'term2': squeeze_tensor(b * d2q_dy2),
        'term3': squeeze_tensor(c * q_pred + d),

        'a': a.item() if torch.is_tensor(a) else a, # Handle scalar tensors
        'b': b.item() if torch.is_tensor(b) else b,
        'c': c.item() if torch.is_tensor(c) else c,
        'd': d.item() if torch.is_tensor(d) else d,
        'epoch': epoch
    })


# --- 全局常量和配置 ---
GLOBAL_SEED = 42
DATA_PATH = "all.xls"
BASE_SAVE_DIR_OPTIMIZATION = r"D:\matlab\crd4\pythondata\BayesianOpt_R2_RESULTS"
EI_ORIG = 26076e6
KGA_ORIG = 421233600
NUM_EPOCHS_FOR_EVAL = 1501 # <--- 迭代次数

# --- 预加载数据和 Scalers ---
set_seed(GLOBAL_SEED)
print("正在加载和预处理数据 (仅一次)...")
train_data, scalers = load_and_preprocess(DATA_PATH)
if train_data is None:
    print("数据加载失败，退出。")
    exit()
# 提取数据备用
X_train = train_data['X_train']
w_train = train_data['w_train']
X_val = train_data['X_val']
w_val = train_data['w_val']
print("数据和 Scaler 加载完成。")

# --- 定义超参数搜索空间  ---
search_space = [
    Real(0.001, 1.0, name='k1'),
    Real(0.001, 1.0, name='k2'),
    Real(1, 10000.0, name='k3'),
    Real(500.0, 5000.0, name='k4'),
    Real(0, 8000.0, name='k5')
]

# --- 定义评估函数 (计算 -R2_val) ---
@use_named_args(search_space)
def evaluate_hyperparameters(**params):

    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    k4 = params['k4']
    k5 = params['k5']

    eval_id = time.strftime("%Y%m%d_%H%M%S") # 使用时间戳作为唯一标识

    print(f"\n--- 开始评估 ID: {eval_id} ---")
    print(f"  超参数: k1={k1:.3e}, k2={k2:.3e}, k3={k3:.3f}, k4={k4:.3f}, k5={k5:.3f}")

    # a. 构建 config
    config = {
        "EI": EI_ORIG * k1, "KGA": KGA_ORIG * k2,
        "lambda_pde": k3, "sig_q": k4, "mu_q": k5,
    }

    # b. 创建本次评估的独立保存目录
    current_save_dir = os.path.join(BASE_SAVE_DIR_OPTIMIZATION, f"eval_{eval_id}")
    os.makedirs(current_save_dir, exist_ok=True)

    # c. 调用训练函数
    current_seed = int(time.time()) # 使用当前时间作为随机种子，避免评估间相互影响
    set_seed(current_seed)

    final_r2_val = -np.inf # 初始化为一个很差的值
    trained_model = None  # 初始化 trained_model

    try:
        # --- 调用 run_training，它会完成训练并保存内部结果，并返回模型 ---
        trained_model = run_training(config, train_data, scalers, current_save_dir,
                                     num_epochs=NUM_EPOCHS_FOR_EVAL)

        # 保存本次评估的模型权重 ---
        if trained_model:  # 确保 run_training 返回了模型
            weights_save_path = os.path.join(current_save_dir, 'model_weights.pth')  # 固定文件名
            torch.save(trained_model.state_dict(), weights_save_path)
            print(f"  模型权重已保存至: {weights_save_path}")
        # --- 保存权重结束 ---

        # --- 从 run_training 保存的文件中读取 R2_val ---
        param_file_path = os.path.join(current_save_dir, 'model_params.xlsx')  # 假设文件名固定
        if os.path.exists(param_file_path):
            results_df = pd.read_excel(param_file_path)
            if not results_df.empty and 'R2_val' in results_df.columns:
                final_r2_val = results_df['R2_val'].iloc[0]  # 读取 R2_val
            else:
                print(f"  警告: {param_file_path} 文件为空或缺少 R2_val 列。")
                # final_r2_val 保持 -np.inf
        else:
            print(f"  警告: 未找到 {param_file_path} 文件。")
            # final_r2_val 保持 -np.inf

        print(f"  评估 ID {eval_id} 完成: R2_val={final_r2_val:.4f}")

    except Exception as e:
        print(f"!!! 评估 ID {eval_id} 失败: {e}")
        # final_r2_val 保持 -np.inf

    # 返回负的 R²，因为 gp_minimize 是最小化
    return -final_r2_val


# --- 主程序：运行贝叶斯优化 ---
if __name__ == "__main__":
    print("\n--- 开始贝叶斯优化 (最小化 -R2_val) ---")

    # 创建优化结果保存目录
    os.makedirs(BASE_SAVE_DIR_OPTIMIZATION, exist_ok=True)

    # --- 运行高斯过程最小化 ---
    n_calls = 10 # <<<=== 总共评估多少组超参数 (包括初始点)
    n_initial_points = 2 # <<<=== 初始随机探索的点数

    print(f"总评估次数: {n_calls}, 初始随机点数: {n_initial_points}")

    result = gp_minimize(
        func=evaluate_hyperparameters, # 评估函数
        dimensions=search_space,       # 超参数空间
        acq_func="EI",                 # Acquisition function (Expected Improvement)
        n_calls=n_calls,               # 总评估次数
        n_initial_points=n_initial_points, # 初始随机点
        random_state=GLOBAL_SEED       # 控制 skopt 内部的随机性
    )

    # --- 分析结果 ---
    print("\n--- 贝叶斯优化完成 ---")

    best_params_list = result.x
    best_neg_r2 = result.fun

    print(f"找到的最佳超参数组合 (k1, k2, k3, k4, k5):")
    print(f"  k1 = {best_params_list[0]:.4f}")
    print(f"  k2 = {best_params_list[1]:.4f}")
    print(f"  k3 = {best_params_list[2]:.4f}")
    print(f"  k4 = {best_params_list[3]:.4f}")
    print(f"  k5 = {best_params_list[4]:.4f}") # Corrected index for k5, it's the 5th element (index 4)
    print(f"对应的最小 (-R2_val): {best_neg_r2:.4f}")
    print(f"对应的最佳 R2_val: {-best_neg_r2:.4f}")

    # 保存优化过程的详细信息
    optimization_log = []
    param_names = [dim.name for dim in search_space]
    for params_iter, neg_r2 in zip(result.x_iters, result.func_vals): # Renamed params to params_iter
        log_entry = {name: val for name, val in zip(param_names, params_iter)}
        log_entry['neg_R2_val'] = neg_r2
        log_entry['R2_val'] = -neg_r2
        optimization_log.append(log_entry)

    log_df = pd.DataFrame(optimization_log)
    log_save_path = os.path.join(BASE_SAVE_DIR_OPTIMIZATION, "bayesian_optimization_log.xlsx")
    log_df.to_excel(log_save_path, index=False)
    print(f"优化过程日志已保存至: {log_save_path}")

    print("\n优化过程结束。最佳参数已打印，详细日志已保存。")
    print(f"每次评估的详细训练结果保存在 {BASE_SAVE_DIR_OPTIMIZATION} 下的 eval_xxx 文件夹中。")