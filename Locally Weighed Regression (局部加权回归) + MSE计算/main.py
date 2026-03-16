# Locally Weighed Regression (局部加权回归) + MSE计算
import numpy as np
from matplotlib import pyplot as plt


# 1. 算法核心：局部加权回归（带正则化，避免矩阵不可逆）
def local_weight_LR(test_point, train_X, train_Y, k=1.0, reg=1e-6):
    xMat = np.array(train_X)
    yMat = np.array(train_Y)
    N, D = np.shape(xMat)

    # 确保测试点维度正确
    test_point = np.array(test_point).reshape(1, -1)
    # 计算样本与测试点的差值
    diff_mat = np.tile(test_point, [N, 1]) - xMat
    # 高斯核权重（正确公式）
    weights = np.exp(-np.sum(diff_mat ** 2, axis=1) / (2 * k ** 2))
    weights = np.diag(weights)  # 对角权重矩阵

    # 添加正则化项，避免矩阵奇异
    xTx = xMat.T @ (weights @ xMat) + reg * np.eye(D)
    ws = np.linalg.inv(xTx) @ xMat.T @ weights @ yMat

    return float(test_point @ ws)


# 2. 批量预测函数
def test_local_weight_LR(test_point, train_X, train_Y, k=1.0):
    N, D = test_point.shape
    Y_hat = np.zeros((N, 1))
    for i in range(N):
        Y_hat[i] = local_weight_LR(test_point[i], train_X, train_Y, k)
    return Y_hat


# 3. 新增：计算最小均方误差（MSE）
def compute_MSE(Y_hat, Y_real):
    """
    计算均方误差：MSE = 1/N * Σ(Y_real - Y_hat)²
    :param Y_hat: 预测值 (N,1)
    :param Y_real: 真实值 (N,1)
    :return: 均方误差（标量）
    """
    Y_hat = np.array(Y_hat)
    Y_real = np.array(Y_real)
    mse = np.sum((Y_real - Y_hat) ** 2) / Y_hat.shape[0]
    return round(mse, 6)  # 保留6位小数，便于查看


# 4. 数据加载函数（适配ex0.txt，纯文件读取）
def load_DataSet(file_path, col_X=1, col_Y=2, add_bias=True):
    try:
        # 读取整个数据文件
        all_data = np.loadtxt(
            file_path,
            dtype=float,
            delimiter=None,
            encoding='utf-8'
        )
        print(f"✅ 成功读取{file_path}：共{len(all_data)}行，{all_data.shape[1]}列")

        # 提取有效特征和标签
        data_X = all_data[:, col_X].reshape(-1, 1)
        data_Y = all_data[:, col_Y].reshape(-1, 1)

        # 添加偏置列
        if add_bias:
            data_X = np.hstack([np.ones((len(data_X), 1)), data_X])

        return data_X, data_Y

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 未找到文件：{file_path}\n请确认文件和代码同目录")
    except ValueError as e:
        raise ValueError(f"❌ 数据格式错误：{e}\n请确认每行3个数字，空格分隔")


if __name__ == "__main__":
    # ========== 1. 加载数据 ==========
    try:
        X, Y = load_DataSet("ex0.txt", add_bias=True)
    except Exception as e:
        print(e)
        exit(1)

    # ========== 2. 划分训练/测试集（可选，也可全量测试） ==========
    # 按8:2划分训练集和测试集（适配你的数据量）
    split_idx = int(len(X) * 0.8)
    train_X, train_Y = X[:split_idx], Y[:split_idx]
    test_X, test_Y = X[split_idx:], Y[split_idx:]
    print(f"\n📊 数据划分：训练集{len(train_X)}条，测试集{len(test_X)}条")

    # ========== 3. 测试不同k值 + 计算MSE ==========
    k_list = [0.1, 0.01, 0.003, 10]  # 新增k=10，完整计算所有测试k值的MSE
    mse_results = {}  # 存储每个k值的MSE

    for k in k_list:
        # 用训练集拟合，测试集预测
        Y_hat = test_local_weight_LR(test_X, train_X, train_Y, k=k)
        # 计算MSE
        mse = compute_MSE(Y_hat, test_Y)
        mse_results[k] = mse
        print(f"🔍 k={k} → 测试集MSE = {mse}")

    # ========== 4. 全量数据拟合 + 绘图（修复子图数量/标题错误） ==========
    Y_hat_1 = test_local_weight_LR(X, X, Y, k=0.1)
    Y_hat_2 = test_local_weight_LR(X, X, Y, k=0.01)
    Y_hat_3 = test_local_weight_LR(X, X, Y, k=0.003)
    Y_hat_4 = test_local_weight_LR(X, X, Y, k=10)

    # 排序使曲线平滑
    index = np.argsort(X[:, 1])
    X_copy = X[index, :]

    # 核心修复：将子图布局改为4行1列，适配4个k值的子图
    fig = plt.figure(figsize=(10, 15))  # 调整画布高度，适配4个子图
    fig.subplots_adjust(hspace=0.6)

    # 子图1：k=0.1 + MSE标注（修复英文标注）
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.scatter(X[:, 1], Y, s=20, alpha=0.8, label="Original Data")
    ax1.plot(X_copy[:, 1], Y_hat_1[index], color="red", linewidth=2,
             label=f"Fitted Curve (MSE={mse_results[0.1]})")
    ax1.set_title(f"k=0.1 (Underfitting) | MSE={mse_results[0.1]}")
    ax1.set_xlabel("X Feature")
    ax1.set_ylabel("Y")
    ax1.legend()

    # 子图2：k=0.01 + MSE标注
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.scatter(X[:, 1], Y, s=20, alpha=0.8, label="Original Data")
    ax2.plot(X_copy[:, 1], Y_hat_2[index], color="red", linewidth=2,
             label=f"Fitted Curve (MSE={mse_results[0.01]})")
    ax2.set_title(f"k=0.01 (Moderate Fitting) | MSE={mse_results[0.01]}")
    ax2.set_xlabel("X Feature")
    ax2.set_ylabel("Y")
    ax2.legend()

    # 子图3：k=0.003 + MSE标注
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.scatter(X[:, 1], Y, s=20, alpha=0.8, label="Original Data")
    ax3.plot(X_copy[:, 1], Y_hat_3[index], color="red", linewidth=2,
             label=f"Fitted Curve (MSE={mse_results[0.003]})")
    ax3.set_title(f"k=0.003 (Overfitting) | MSE={mse_results[0.003]}")
    ax3.set_xlabel("X Feature\n\n")
    ax3.set_ylabel("Y")
    ax3.legend()

    # 子图4：k=10 + MSE标注（修复标题错误，匹配k=10的MSE）
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.scatter(X[:, 1], Y, s=20, alpha=0.8, label="Original Data")
    ax4.plot(X_copy[:, 1], Y_hat_4[index], color="red", linewidth=2,
             label=f"Fitted Curve (MSE={mse_results[10]})")
    ax4.set_title(f"k=10 (Severe Underfitting) | MSE={mse_results[10]}")
    ax4.set_xlabel("X Feature")
    ax4.set_ylabel("Y")
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # ========== 5. 结果总结 ==========
    print("\n📈 结果总结：")
    min_k = min(mse_results, key=mse_results.get)
    print(f"✅ MSE最小的k值：{min_k}，对应的MSE={mse_results[min_k]}")
    max_k = max(mse_results, key=mse_results.get)
    print(f"❌ MSE最大的k值：{max_k}，对应的MSE={mse_results[max_k]}")
