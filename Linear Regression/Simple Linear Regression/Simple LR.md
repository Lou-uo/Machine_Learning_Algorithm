Simple Linear Regression    简单线性回归

<img src="D:\Table\Big%20Files\Machine_Learning_Algorithm\Linear%20Regression\Simple%20Linear%20Regression\Day%202.jpg" alt="Day 2" style="zoom: 67%;" />

**Simple Linear Regression（简单线性回归）**，是机器学习里最基础的监督学习算法之一。

核心目标是：

> 用一个变量 $X$ 去预测另一个变量 $Y$



**Predicting a response using a single feature.**

仅使用单个输入变量 (特征) 来预测结果
$$
y = b_0 + b_1x_1
$$

- $y$：预测值 (因变量 / 输出)
- $x_1$：输入特征 (自变量)
- $b_0$：截距 (intercept)
- $b_1$：斜率 (slope)



E.g. 

$Score = b_0 + b_1 \times hours$

表示为 `学生的成绩 = 截距 + 学习时间 × 权重 `



==关键问题：== **如何找到最佳的拟合线**

> 找到一条 “最佳拟合线”，使得预测误差最小

- 预测误差：$y_i - y_p$ ，即 `真实值 - 预测值`

- 目标函数：$\min \{\sum(y_i-y_p)^2\}$，使得 `让所有误差平方和最小`，即 **最小二乘法 (Least Squares)**



---



==算法流程：==



#### STEP 1：Preprocess the Data

数据预处理阶段

- 导入库（Import Libraries）

- 导入数据集（Import Dataset）

- 检查缺失值（Missing Data）

- 划分训练集和测试集（Split Dataset）

- 特征缩放（Feature Scaling）



#### STEP 2：Fitting Simple Linear Regression Model to the Training Set

使用 `Scikit-Learn` 机器学习库 【需要配置 `pip install scikit-learn`】

```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
```

- 创建回归模型
- 用训练数据训练



 #### STEP 3：Predicting the Result

使用训练好的模型预测测试集结果

```python
y_pred = regressor.predict(X_test)
```



#### STEP 4：Visualization

使用 matplotlib 绘制散点图和回归线 【需要配置 `pip install matplotlib`】

```python
plt.scatter(X, y)
plt.plot(X, y_pred)
```

- 看预测效果好不好

- 看数据是否接近直线



---



#### Code

```python
# Step 1: Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pathlib import Path

csv_path = Path(__file__).parent / 'studentscores.csv'
dataset = pd.read_csv(csv_path)
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

# Step 2: Fitting Simple Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# Step 3: Predecting the Result
Y_pred = regressor.predict(X_test)

# Step 4: Visualization
# Visualising the Training results
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()
# Visualizing the Test results
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()
```



#### Results

![Figure_1](D:\Table\Big Files\Machine_Learning_Algorithm\Linear Regression\Simple Linear Regression\Figure_1.png)

**训练集可视化**

- <span style="color:red">红色散点</span>：训练数据 (用于训练模型的样本)
- <span style="color:blue">蓝色直线</span>：线性回归模型在训练集上的拟合线
- 横轴：学习时间
- 纵轴：考试分数

![Figure_2](D:\Table\Big Files\Machine_Learning_Algorithm\Linear Regression\Simple Linear Regression\Figure_2.png)

**测试集可视化**

- <span style="color:red">红色散点</span>：测试数据（未参与模型训练，用于评估模型）
- <span style="color:blue">蓝色直线</span>：同样的线性回归拟合线

- 测试数据点较少（因为测试集只占25%）
- 测试点分布在2小时到7.5小时之间
- 所有测试点都比较接近回归线，说明模型泛化能力较好
