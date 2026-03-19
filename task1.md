### 1. 学习雅可比矩阵
把偏导堆在一起，利用 $\frac{d\text{位置}}{dt} = v = \sum (\text{位置函数偏导} \times \text{对应变量变化率})$，快速求速度分运动矩阵。

---

### 2. 拉格朗日力学
（思考：引入拉格朗日力学是因为在关节较多即变量较多时，用牛顿力学受力分析极难，而拉格朗日力学则可以由能量直接推出运动方程 $\dot{x} = Ax + Bu$）

理解了拉格朗日量（动能减去势能）表示了整个系统的活跃程度，并且物体运动会选择一个作用量最小的路线。
（思考：大自然喜欢把能量以势能的形式存储，有时候要考虑非保守力）。
作用：用于求物体运动方程。

利用雅可比矩阵可以在已知作用器末端与关节角的关系时，用于写出末端的速度。
速度则用于计算拉格朗日量，并代入拉格朗日方程，进而得到运动方程 $\dot{x} = Ax + Bu$（获得 $\dot{x} = Ax + Bu$ 的过程不太清楚）。

---

### 3. 控制干预：PID、LQR、MPC
知道如何获得运动方程后就可以去学会如何干预运动，用三种控制：

#### PID
* **proportional 比例控制**：只看当下误差，误差越大用力越大。
* **integral 积分**：累积误差越大用力越大，用于最后矫正因一些因素一直未回到平衡点的情况。
* **derivative 微分**：求当前变化快慢，如果变化过快，则会负责踩下刹车。
*(代码部分较简单)*

---

#### LQR (Linear Quadratic Regulator)
学习过程：（把控制过程比喻为下山）

1. 理解到物理世界运行虽然大多为非线性系统，但计算机擅长计算矩阵运算，故常把各系统近似为线性的。而运动系统的线性表述多用状态空间表达式 $\dot{x} = Ax + Bu$。这个式子的意思是：位置的变化率等于 $A$（物理规律）点乘物理状态加上 $B$（控制输入矩阵：描述输入会怎么样影响系统）点乘 $u$（控制力/输入）。
2. 引入代价函数 $J$：
   $$J = \sum_{k=0}^{\infty} (x_k^T Q x_k + u_k^T R u_k)$$
   （左乘转置右乘正置形成二次型，怎么样都是正的）。损失函数有两个重要的参数 $Q, R$。$Q$ 可以比喻为对误差的厌恶度，$R$ 可以形容为对能量使用的厌恶度（$x$ 是空间表达式，$u$ 是控制量）。
3. 价值函数 $V$：若 $J$ 是用来评价什么是好。则 $V$ 是用于给每一个 $x$ 打分：若在 $x$ 处表现完美，则下山的代价就是 $J$。这看上去似乎 $V$ 就和 $J$ 没什么区别？
4. 转化：$J$ 是无穷积分，计算机不好处理。那就假设 $V(x)$ 可以写成 $x^T S x$，即把 $J$ 里面的 $A, B, R, Q$ 压缩为一个矩阵 $S$，用黎卡提方程解（不懂）。
5. $S$ 到 $K$ 反馈矩阵（救生指南）的转化。（为什么不用 $V$ 得到 $K$？因为 $S$ 矩阵相当于整座山，得到一个 $x$ 可以映射为海拔。$V(x)$ 只是某个海拔）。
   $K = R^{-1}(B^T)S$
   * **$S$（全局视野）**：它告诉你如果现在不纠偏，未来会付出多大的代价。$S$ 越大，说明未来越危险，控制力就得越大。
   * **$B^T$（转换器）**：$B$ 是你的控制力如何影响状态，它的转置 $B^T$ 则反过来告诉你，为了改变未来的状态，现在的控制力应该往哪个方向使。（那为什么不是 $B$ 的逆？）
   * **$R^{-1}$（预算门槛）**：这是最关键的。$R$ 是你对电费的在意程度。

**代码解析：**
```python
def control_output(self, state_des, state_now):
    # 在类内部的函数
    # 传入 state_des 是 desire 的意思，即预测值。now 即当前值

    S = np.matrix(linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
    # 跳过求 J 的过程传入类的属性 A,B,Q,R 铁锅乱炖，求出 S（“对整座山的整体描述”）

    K = np.matrix(linalg.inv(self.B.T * S * self.B + self.R) * (self.B.T * S * self.A))
    # 数学解释：inv 是求逆的意思，在矩阵运算中左乘逆矩阵相当于除以这个矩阵
    # 故意思是 (self.B.T * S * self.A) / (self.B.T * S * self.B + self.R)

    '''
    左半部分：linalg.inv(B.T * S * B + R) 我们可以叫它【动作的阻力/代价】
    B.T * S * B：B 是你的控制能力（比如方向盘转角），S 是未来的代价。这一项的意思是：如果你现在做一个动作 B，它会对未来产生的“总成本”造成多大的扰动。如果 S 很大，说明你这个动作稍微动一点，未来的风险就会变动很大。
    R：这就是你说的“电费”或“体力的厌恶程度”。
    加和再求逆（inv）：把“未来的风险”和“当下的电费”加在一起，就是你做一个动作的总代价。求逆（1/x 的效果）的意思是：总代价越高，你的控制增益 K 就得越小。
    通俗点说：如果电费很贵（R大）或者乱动会导致未来很危险（S大），那你的 K 就要变小，动作要变得极其保守。

    公式右半部分：(B.T * S * A) 我们可以叫它【纠偏的紧迫性】
    A：这是物理规律，代表如果你现在不收手，下一刻系统会自己惯性滑行到哪里。
    S * A：这一项是在问：按照物理规律 A 走下去，未来会欠下多少“债”（S）。
    B.T * (S * A)：这是在计算：我的控制力 B，在多大程度上能抵消掉由于 A 带来的未来亏损。它决定了你控制的方向和力度。如果 B 刚好能抵消掉 A 带来的坏处，这一项的值就会很大，引导 K 去做一个强力的反馈。
    '''

    # print("state_now", state_now)
    # print("state_des", state_des)
    
    action_list = -10 * np.dot(K, state_now - state_des)
    # 利用指南，算出动作矩阵

    action = action_list[0,0]
    # 动作矩阵只有左上角那个是有效的

    return action
```
  
#### MPC (Model Predictive Control)

如果 LQR 是在运动开始就算好了整个地图，MPC 就是走一步看 $N$ 步的预测模型。相对于 LQR 引入了 $N$ 步长、$Q$ 状态权重矩阵、$R$ 控制输入权重矩阵、$F$ 终端状态权重矩阵（担心系统只把前 $N$ 步算得完美，用 $F$ 限制终端状态）。
其他部分和 LQR 相似。

**代码部分：**：
    # 求出 E、H 矩阵
def get_QPMatrix(A, B, Q, R, F, N):
    M = np.vstack([np.eye(n), np.zeros((N*n, n))])
    # np.vstack：vertical stack 
    # np.zeros((N*n, n)) 生成行列数对应参数的零矩阵。即把几个 np 数组竖直堆起来成矩阵。此处单位矩阵与零矩阵堆叠

    C = np.zeros(((N+1)*n, N*p))
    # 这两步初始化 M、C 矩阵

    # 接下来把 X 和 U 拍成平面
    temp = np.eye(n)
    for i in range(1, N+1):
        rows = i * n + np.arange(n)
        C[rows,:] = np.hstack([temp @ B, C[rows-n, :-p]])  # horizontal stack横向堆叠
        temp = A @ temp
        M[rows,:] = temp
    
    '''
    M 是系统的惯性（不出力自己怎么动）。C 是你的控制影响力。
    M 是 A（物理规律）的幂次堆叠。C 是 u 的控制的叠加。
    在 MPC 的数学表达中，为了方便一次性计算，我们会把未来 N 步的所有状态垂直堆叠成一个超级列向量：
    X = M x_k + C U
    然后可以代入损失函数 J 就变成二次型：1/2 (U^T) H U + (U^T) E x_k
    '''

    Q_ = np.kron(np.eye(N), Q)
    rows_Q, cols_Q = Q_.shape
    rows_F, cols_F = F.shape

    Q_bar = np.zeros((rows_Q+rows_F, cols_Q+cols_F))
    Q_bar[:rows_Q, :cols_Q] = Q_
    Q_bar[rows_Q:, cols_Q:] = F
    
    R_bar = np.kron(np.eye(N), R)

    # G = M.T @ Q_bar @ M
    E = C.T @ Q_bar @ M
    H = C.T @ Q_bar @ C + R_bar
    
    return E, H

def mpc_prediction(x_k, E, H, N, p):
    # 定义优化变量
    U = ca.SX.sym('U', N * p)
    
    # 定义目标函数
    objective = 0.5 * ca.mtimes([U.T, H, U]) + ca.mtimes([U.T, E, x_k])
    
    qp = {'x': U, 'f': objective}
    opts = {'print_time': False, 'ipopt': {'print_level': 0}}
    solver = ca.nlpsol('solver', 'ipopt', qp, opts)

    # 求解问题
    sol = solver()
    
    # 提取最优解
    U_k = sol['x'].full().flatten()
    u_k = 10 * U_k[:p]  # 取第一个结果

    return u_k