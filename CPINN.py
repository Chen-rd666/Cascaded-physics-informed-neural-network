import torch
import torch.nn as nn
import torch.autograd as autograd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas import  ExcelWriter
import os


# -------------------- 模型定义 --------------------
# 网络Q：计算荷载
class QNetwork(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)  # 输出q
        )

    def forward(self, x):
        return self.encoder(x)

# 网络W：输入q + 位置y，计算变形
class WNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),  # 输入维度：q (1) + y (1)
            nn.Tanh(),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)  # 输出w
        )

    def forward(self, q, y):
        input = torch.cat([q, y], dim=1)
        return self.encoder(input)

# 联合模型（级联A+B）与物理约束
class CascadedPINN(nn.Module):
    def __init__(self, config, scalers):
        super().__init__()
        # 物理信息超参数
        self.EI = nn.Parameter(torch.tensor(config["EI"]), requires_grad=False)
        self.KGA = nn.Parameter(torch.tensor(config["KGA"]), requires_grad=False)
        self.lambda_pde = nn.Parameter(torch.tensor(config["lambda_pde"]), requires_grad=False)
        self.sig_q = nn.Parameter(torch.tensor(config["sig_q"], dtype=torch.float32),requires_grad=False)
        self.mu_q = nn.Parameter(torch.tensor(config["mu_q"], dtype=torch.float32),requires_grad=False)

        # 从 scalers['X'] 中获取第9列参数（y的归一化参数）
        self.sig_y = nn.Parameter(
            torch.tensor(scalers['X']['scale'][-1], dtype=torch.float32),  # 最后一列为y
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
        self.q_net = QNetwork()
        self.w_net = WNetwork()

    def forward(self, x):
        # 输入x包含推进参数和位置y
        y = x[:, -1:]  # 提取位置y
        q = self.q_net(x)
        w = self.w_net(q, y)
        return q, w

    def compute_q_derivatives(self,x,order=2):
        # 计算网络A中 q 对 y 的导数
        q = self.q_net(x)
        derivative = [q]   # 初始时包含零阶导数（即原始输出q）
        current_derivation = q   # 当前导数变量初始化为q

        for _ in range(order):
            # 计算当前导数对y的梯度
            grad = autograd.grad(
                outputs=current_derivation,
                inputs=x,
                grad_outputs=torch.ones_like(current_derivation),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            grad_y = grad[:, -1:]  # 取最后一列（y方向梯度）

            derivative.append(grad_y)
            current_derivation = grad_y

        return tuple(derivative)

    def compute_w_derivatives(self,x,order=4):
        '''计算网络B中 w 对 y 的导数（支持四阶导数）'''
        # 切断q对y的求导
        with torch.no_grad():
            q = self.q_net(x)

        y = x[:,-1:].requires_grad_(True)
        w = self.w_net(q,y)

        derivative = [w]
        current_derivation = w
        for _ in range(order):
            # 计算当前导数（对 y 求导）
            grad = autograd.grad(
                outputs=current_derivation,
                inputs=y,
                grad_outputs=torch.ones_like(current_derivation),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            derivative.append(grad)
            current_derivation = grad

        return tuple(derivative)  # (w, dw/dy, d²w/dy², d³w/dy³, d⁴w/dy⁴)

    def compute_loss(self, x, w_true):
        # 计算网络Q的 q 及其导数
        q_pred, dq_dy, d2q_dy2 = self.compute_q_derivatives(x, order=2)

        # 计算网络W的 w 及其导数
        w_pred, dw_dy, d2w_dy2, d3w_dy3, d4w_dy4 = self.compute_w_derivatives(x, order=4)

        # PDE残差项系数计算
        a = self.sig_w / (self.sig_y ** 4)
        b = self.sig_q / (self.sig_y ** 2) * (1 / self.KGA)
        c = self.sig_q / self.EI
        d = self.mu_q / self.EI
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
        return (q_pred.detach(),    # q的预测值
                dq_dy.detach(),     # 一阶导
                d2q_dy2.detach(),   # 二阶导
                w_pred.detach(),    # w的预测值
                dw_dy.detach(),     # 一阶导
                d2w_dy2.detach(),   # 二阶导
                d3w_dy3.detach(),   # 三阶导
                d4w_dy4.detach(),   # 四阶导
                a,                  # self.sig_w / (self.sig_y**4)
                b,                  # self.sig_q / (self.sig_y**2) * (1/self.KGA)
                c,                  # self.sig_q / self.EI
                d,                  # self.mu_q / self.EI
                pde_loss,           # 偏微分损失
                data_loss_w,        # w数据损失
                total_loss         # 总损失
                )


# -------------------- 数据预处理 --------------------
def load_and_preprocess(data_path, w_scale=0.001, train_ratio=0.8, random_state=10):
    # 加载数据，合并推进参数和位置y作为输入特征，并进行归一化
    data = pd.read_excel(data_path)
    np.random.seed(random_state)
    # 提取特征：假设前2-15列为推进参数，最后两列为位移w和位置y
    X_features = data.iloc[:, 1:16]  # 前8列推进参数
    y = data.iloc[:, -1].values.reshape(-1, 1)  # 最后一列为y
    displacement = data.iloc[:, -2].values.reshape(-1, 1) * w_scale  # 位移w转换为米

    # 合并推进参数和y作为输入X，共16维特征（15+1）
    X = pd.concat([X_features, pd.DataFrame(y, columns=['y'])], axis=1)

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

    # -------------------- 数据分割 --------------------
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


# -------------------- 训练函数 --------------------
def run_training(config, train_data, scalers, save_dir):
    # 初始化
    model = CascadedPINN(config, scalers)
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
    for epoch in range(1501):
        model.train()
        optimizer.zero_grad()

        # 前向传播（仅训练集）
        q_pred, w_pred = model.forward(X_train)

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
            "sig_q": config["sig_q"],       # 新增参数
            "mu_q": config["mu_q"],         # 新增参数
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
            # 使用enable_grad以计算梯度
            with torch.enable_grad():
                X_val = train_data['X_val'].requires_grad_(True)
                q_val_pred, w_val_pred = model(X_val)
                # 重新计算验证损失和导数
                (q_val_pred, dq_dy_val, d2q_dy2_val,
                 w_val_pred, dw_dy_val, d2w_dy2_val,
                 d3w_dy3_val, d4w_dy4_val,
                 a_val, b_val, c_val, d_val,
                 pde_loss_val, data_loss_w_val,
                 total_loss_val) = model.compute_loss(
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

    # 最终保存三张表
    # 表1：损失历史
    pd.DataFrame(loss_history).to_excel(
            os.path.join(save_dir, 'loss_history.xlsx'),
            index=False
            )

    # 表2：所有预测结果（分sheet存储）
    with pd.ExcelWriter(os.path.join(save_dir, 'all_predictions.xlsx')) as writer:
        for i, df in enumerate(pred_history):
            df.to_excel(writer, sheet_name=f'epoch_{i * 100}', index=False)

    # -------------------- R²计算 --------------------
    # 计算R²函数
    def compute_r2(y_true, y_pred):
        """计算决定系数（支持torch.Tensor输入）"""
        y_true = y_true.cpu().numpy().flatten()
        y_pred = y_pred.cpu().numpy().flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # 最终模型评估
    model.eval()
    with torch.no_grad():
        # 训练集预测
        _, w_pred_train = model(train_data['X_train'])
        # 验证集预测
        _, w_pred_val = model(train_data['X_val'])

    # 计算R²（基于原始物理量纲）
    R2_train = compute_r2(train_data['w_train'], w_pred_train)
    R2_val = compute_r2(train_data['w_val'], w_pred_val)

    # 表3：模型参数
    # -------------------- 修改参数保存代码 --------------------
    param_df = pd.DataFrame([{
        'EI': config['EI'],
        'KGA': config['KGA'],
        'lambda_pde': config['lambda_pde'],
        "sig_q": config["sig_q"],               # 新增参数
        "mu_q": config["mu_q"],                 # 新增参数
        'sig_w': scalers['w']['scale'],
        'sig_y': scalers['X']['scale'][-1],     # 从合并特征中提取y的scale
        'mu_w': scalers['w']['mean'],
        'mu_y': scalers['X']['mean'][-1],       # 从合并特征中提取y的mean
        'R2_train': R2_train,
        'R2_val': R2_val
    }])

    print(R2_train, R2_val)
    param_df.to_excel(os.path.join(save_dir, 'model_params.xlsx'), index=False)

    return model


# -------------------- 预测数据生成函数 --------------------
def create_prediction_df(epoch, X,
                         q_pred, dq_dy, d2q_dy2,
                         w_pred, dw_dy, d2w_dy2,
                         d3w_dy3, d4w_dy4,
                         a, b, c, d,
                         w_true,
                         scalers, suffix):

    # 转换为numpy数组并压缩维度（确保一维）
    def squeeze_tensor(tensor):
        return tensor.detach().cpu().numpy().squeeze()

    return pd.DataFrame({
        'Dataset': [suffix] * len(X),
        'y': squeeze_tensor(X[:, -1]),  # 直接取归一化后的y值

        # q相关项
        'pred_q': squeeze_tensor(q_pred),
        'dq/dy': squeeze_tensor(dq_dy),
        'd2q/dy2': squeeze_tensor(d2q_dy2),

        # w相关项
        'pred_w': squeeze_tensor(w_pred),
        'true_w': squeeze_tensor(w_true),  # 保持归一化后的值
        'dw/dy': squeeze_tensor(dw_dy),
        'd2w/dy2': squeeze_tensor(d2w_dy2),
        'd3w/dy3': squeeze_tensor(d3w_dy3),
        'd4w/dy4': squeeze_tensor(d4w_dy4),

        # PDE项
        'term1': squeeze_tensor(a * d4w_dy4),
        'term2': squeeze_tensor(b * d2q_dy2),
        'term3': squeeze_tensor(c * q_pred + d),

        # 参数（标量自动广播）
        'a': a.item(),
        'b': b.item(),
        'c': c.item(),
        'd': d.item(),
        'epoch': epoch
    })

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 定义原始参数常量
    EI_ORIG = 26076e6  # 添加原始值定义
    KGA_ORIG = 421233600

    # 数据加载（通过函数调用）
    train_data, scalers = load_and_preprocess("all.xls")  # 传入数据路径

    # 读取超参数组合
    hypa_df = pd.read_excel(r"C:\Users\瓜皮儿子\Desktop\123\hypa.xlsx", header=None)
    param_combinations = hypa_df.values
    # 遍历所有参数组合（需要解包5个参数）
    for idx, (k1, k2, k3, k4, k5) in enumerate(param_combinations, 1):  # 解包5个参数
        # 创建参数文件夹
        folder_name = f"{idx}_EI={k1:.1e}_KGA={k2:.1e}_lambda={k3:.1f}_sigq={k4:.2f}_muq={k5:.2f}"
        save_dir = os.path.join(r"C:\Users\瓜皮儿子\Desktop\123", folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # 计算实际参数
        config = {
            "EI": EI_ORIG * k1,
            "KGA": KGA_ORIG * k2,
            "lambda_pde": k3,
            "sig_q": k4,
            "mu_q": k5
        }

        # 打印信息
        print(f"\n▶ 正在训练组合 {idx}/{len(param_combinations)}")
        print(f"      EI: {k1:.3e} kN·m²")
        print(f"     KGA: {k2:.3e} kN/m")
        print(f"  Lambda: {k3:.3f}")
        print(f" Sigma_q: {k4:.3f}")
        print(f"    Mu_q: {k5:.3f}")



        # 执行训练并保存结果
        trained_model = run_training(config, train_data, scalers, save_dir)

        # 保存模型权重
        torch.save(trained_model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

    print("\n✅ 所有参数组合训练完成！结果已保存至对应文件夹")