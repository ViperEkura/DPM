## DDPM

### 1. 前向扩散过程

给定真实数据样本 $x_0 \sim q(x_0)$，DDPM 定义一个固定的、参数化的**马尔可夫链**，在 $T$ 步内将数据逐渐转化为标准高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t I), \quad t = 1, 2, \dots, T
$$

其中：
- $\beta_t \in (0, 1)$ 是预设的小方差（通常随 $t$ 缓慢增大）
- 整个前向过程的联合分布为：

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
$$

定义累积量：

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

则可以**直接采样任意时刻 $x_t$**（重参数化技巧）：

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) I)
$$

即：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$


### 2. 反向生成过程

目标是学习一个**可学习的马尔可夫链** $p_\theta$ 来逆转前向过程：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \, \Sigma_\theta(x_t, t))
$$

在 DDPM 中，通常**固定方差**（例如 $\Sigma_\theta(x_t, t) = \sigma_t^2 I$，其中 $\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \tilde{\beta_t}$ ），只学习均值 ${\mu}_{\theta}$。

关键洞察：利用贝叶斯规则和高斯性质，可以推导出**真实后验** $q(x_{t-1} \mid x_t, x_0)$ 也是高斯分布：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \, \tilde{\beta}_t I)
$$

其中：

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

由于 $x_0$ 未知，DDPM 训练神经网络 $\epsilon_\theta(x_t, t) $ 来预测前向过程中加入的噪声 $\epsilon$ 。

由 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，可得：

$$
x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon}{\sqrt{\bar{\alpha}_t}}
$$

代入 $\tilde{\mu}_t$，得到用 $\epsilon_\theta$ 表示的均值：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

（这是 DDPM 论文中常用的等价形式）



### 3. 训练目标

DDPM 最小化负对数似然的变分下界（ELBO），但通过重参数化可简化为**噪声预测的均方误差**：

$$
\mathcal{L}_\text{DDPM} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

其中：
- $t \sim \text{Uniform}(\{1, \dots, T\})$
- $x_0 \sim q(x_0)$
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

这个损失函数非常简单且易于优化。



### 4. 采样算法

训练完成后，从纯噪声开始反向生成：

1. 采样 $x_T \sim \mathcal{N}(0, I)$
2. 对 $t = T, T-1, \dots, 1$：
   $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I) \ (\text{若 } t > 1,\ z=0 \text{ 若 } t=1)$
   其中 $\sigma_t^2 = \tilde{\beta_t} $（或设为 $\beta_t$）

