## DDIM

DDIM（Denoising Diffusion Implicit Models）之所以**允许任意时间子序列采样**（即跳步采样，如从 $x_{1000} \to x_{500} \to x_{200} \to x_0$），其根本原因在于：**DDIM 的反向过程不依赖马尔可夫性，而是直接建模任意两个时间步之间的确定性或可控随机映射**，且该映射仅依赖于对原始数据 $x_0$ 的估计。

下面我们结合公式严格解释这一性质。



### 一、核心前提：前向过程的“任意时刻可直达”性质

在扩散模型中，前向过程是**可解析计算任意 $t$ 时刻分布**的：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这意味着，**给定 $x_0$，我们可以直接生成任意 $x_t$，无需经过中间步骤**。  
更重要的是，这个关系是**双向可逆的**（在已知 $\epsilon$ 或能估计 $x_0$ 的前提下）。



### 二、DDIM 的关键洞察：任意两步之间的“一致性路径”

DDIM 不再假设反向过程必须满足：

$$
p_\theta(x_{t-1} | x_t) \text{ 仅依赖 } x_t
$$

（这是马尔可夫假设）

而是考虑更一般的设定：给定当前状态 $x_t$，我们想一步跳到任意更早的时间步 $x_s$（$s < t$），并希望这个跳跃仍然与原始数据分布一致。

为此，DDIM 利用如下事实：

> 如果我们知道（或能估计）真实的 $x_0$，那么 $x_t$ 和 $x_s$ 都是 $x_0$ 的带噪版本，它们之间存在一个**确定性的几何关系**。

具体地，由前向过程：

$$
\begin{aligned}
x_t &= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \\
x_s &= \sqrt{\bar{\alpha}_s} x_0 + \sqrt{1 - \bar{\alpha}_s} \epsilon
\end{aligned}
$$

注意：**同一个 $\epsilon$** 被用于生成所有 $x_t$（这是重参数化的核心）。

于是，我们可以**消去 $x_0$**，得到 $x_s$ 关于 $x_t$ 和 $\epsilon$ 的表达式：

$$
x_s = \sqrt{\frac{\bar{\alpha}_s}{\bar{\alpha}_t}} x_t + \left( \sqrt{1 - \bar{\alpha}_s} - \sqrt{\frac{\bar{\alpha}_s}{\bar{\alpha}_t}} \sqrt{1 - \bar{\alpha}_t} \right) \epsilon
$$

但实际中 $\epsilon$ 未知，我们用神经网络预测它：$\epsilon \approx \epsilon_\theta(x_t, t)$，并由此估计：

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

然后代入 $x_s = \sqrt{\bar{\alpha}_s} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_s} \epsilon_\theta(x_t, t)$，就得到从 $x_t$ **一步跳到 $x_s$** 的更新规则：

$$
\boxed{
x_s = \sqrt{\bar{\alpha}_s} \, \hat{x}_0(x_t, t) + \sqrt{1 - \bar{\alpha}_s - \sigma_{t \to s}^2} \, \epsilon_\theta(x_t, t) + \sigma_{t \to s} z
}
$$

其中 $\sigma_{t \to s} \geq 0$ 控制跳跃中的随机性（当 $\sigma=0$ 时为完全确定性）。

> **关键点**：这个公式只依赖于当前 $x_t$ 和目标时间步 $s$，**不依赖中间任何 $x_{t-1}, x_{t-2}, \dots$**。因此，我们可以自由选择任意子序列 $\{t_0=T, t_1, t_2, \dots, t_N=0\}$ 进行采样。

---

### 三、为什么 DDPM 不能跳步？

DDPM 的反向过程被严格定义为**马尔可夫链**：

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_t)
$$

其推导依赖于**真实后验 $q(x_{t-1} | x_t, x_0)$ 的高斯形式**，而该后验本身是基于**相邻两步**的转移（即 $q(x_t|x_{t-1})$ 和 $q(x_{t-1}|x_0)$）通过贝叶斯法则得到的。

若试图从 $x_t$ 直接跳到 $x_{t-2}$，则：
- 没有对应的 $q(x_{t-2} | x_t, x_0)$ 的简单闭式（除非重新推导）
- 更重要的是，DDPM 的训练目标和方差设计（如 $\tilde{\beta}_t$）**只保证相邻步的 KL 散度最小化**，跨步使用会导致分布偏移

因此，**DDPM 的反向过程不具备跨步一致性**。

