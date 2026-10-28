"""多因子预测模型：LightGBM / XGBoost / GPR / Transformer。

统一接口（`fit` / `predict`），Walk-forward 驱动只认这两个方法，
换模型不用改驱动代码。

## 为什么 Transformer 是手写的

torch 的 CPU 轮子装完 1GB 以上，打进安装包会让三个平台的产物从 ~99MB 涨到
1GB 左右，对一个桌面研究工具来说不可接受。所以这里用 numpy 实现了一个
**微型反向传播 + 单层多头自注意力编码器**（约 250 行），参数规模只有几千，
在 1500 个样本上训练几十轮不到一秒。

它是货真价实的 Transformer 编码器：位置编码 → 多头自注意力 → 残差+LayerNorm
→ 前馈 → 残差+LayerNorm → 取最后一步的表示做线性输出。只是没有 torch 那套
工程封装，也没有 GPU 加速 —— 这个数据量本来也不需要。

## 为什么 GPR 要限制训练样本

高斯过程是 O(n³)（Cholesky 分解）。4000 个样本要几秒，30 折 walk-forward
就是几分钟。这里截取**最近**的 N 个样本训练（默认 400），既能跟上最新市场
状态，又把单折压到几十毫秒。
"""

from __future__ import annotations

import warnings

import numpy as np


# ================================================================== 自动微分
class Var:
    """极简反向传播节点。只支持 Transformer 需要的算子。"""

    __slots__ = ("d", "g", "_bp", "_ch")

    def __init__(self, data, children=()):
        self.d = np.asarray(data, dtype=np.float64)
        self.g = None
        self._bp = None
        self._ch = children


def _acc(old, new):
    return new if old is None else old + new


def _unbroadcast(g, shape):
    """把广播后的梯度还原成原形状。"""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def backward(root: Var) -> None:
    """按拓扑序反向传播，梯度累加到各节点的 `.g`。"""
    order, seen = [], set()

    def visit(v: Var):
        stack = [(v, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for c in node._ch:
                if id(c) not in seen:
                    stack.append((c, False))

    visit(root)
    root.g = np.ones_like(root.d)
    for node in reversed(order):
        if node._bp is not None and node.g is not None:
            node._bp()


def add(a: Var, b: Var) -> Var:
    out = Var(a.d + b.d, (a, b))

    def bp():
        a.g = _acc(a.g, _unbroadcast(out.g, a.d.shape))
        b.g = _acc(b.g, _unbroadcast(out.g, b.d.shape))

    out._bp = bp
    return out


def mul(a: Var, b: Var) -> Var:
    out = Var(a.d * b.d, (a, b))

    def bp():
        a.g = _acc(a.g, _unbroadcast(out.g * b.d, a.d.shape))
        b.g = _acc(b.g, _unbroadcast(out.g * a.d, b.d.shape))

    out._bp = bp
    return out


def matmul(a: Var, b: Var) -> Var:
    out = Var(a.d @ b.d, (a, b))

    def bp():
        ga = out.g @ np.swapaxes(b.d, -1, -2)
        gb = np.swapaxes(a.d, -1, -2) @ out.g
        a.g = _acc(a.g, _unbroadcast(ga, a.d.shape))
        b.g = _acc(b.g, _unbroadcast(gb, b.d.shape))

    out._bp = bp
    return out


def scale(a: Var, k: float) -> Var:
    out = Var(a.d * k, (a,))

    def bp():
        a.g = _acc(a.g, out.g * k)

    out._bp = bp
    return out


def add_const(a: Var, k) -> Var:
    out = Var(a.d + k, (a,))

    def bp():
        a.g = _acc(a.g, _unbroadcast(out.g, a.d.shape))

    out._bp = bp
    return out


def tanh(a: Var) -> Var:
    t = np.tanh(a.d)
    out = Var(t, (a,))

    def bp():
        a.g = _acc(a.g, out.g * (1.0 - t * t))

    out._bp = bp
    return out


def relu(a: Var) -> Var:
    mask = a.d > 0
    out = Var(a.d * mask, (a,))

    def bp():
        a.g = _acc(a.g, out.g * mask)

    out._bp = bp
    return out


def softmax_last(a: Var) -> Var:
    x = a.d - a.d.max(axis=-1, keepdims=True)
    e = np.exp(x)
    s = e / e.sum(axis=-1, keepdims=True)
    out = Var(s, (a,))

    def bp():
        g = out.g
        a.g = _acc(a.g, s * (g - (g * s).sum(axis=-1, keepdims=True)))

    out._bp = bp
    return out


def layernorm(x: Var, gamma: Var, beta: Var, eps: float = 1e-5) -> Var:
    mu = x.d.mean(axis=-1, keepdims=True)
    var = x.d.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x.d - mu) * inv
    out = Var(xh * gamma.d + beta.d, (x, gamma, beta))

    def bp():
        g = out.g
        red = tuple(range(g.ndim - 1))
        gamma.g = _acc(gamma.g, (g * xh).sum(axis=red))
        beta.g = _acc(beta.g, g.sum(axis=red))
        gx = g * gamma.d
        gxhat = gx - gx.mean(axis=-1, keepdims=True) - xh * (gx * xh).mean(
            axis=-1, keepdims=True
        )
        x.g = _acc(x.g, gxhat * inv)

    out._bp = bp
    return out


def take_last(x: Var) -> Var:
    """取序列最后一步：x[:, -1, :]。"""
    out = Var(x.d[:, -1, :], (x,))

    def bp():
        g = np.zeros_like(x.d)
        g[:, -1, :] = out.g
        x.g = _acc(x.g, g)

    out._bp = bp
    return out


def reshape(x: Var, shape) -> Var:
    out = Var(x.d.reshape(shape), (x,))

    def bp():
        x.g = _acc(x.g, out.g.reshape(x.d.shape))

    out._bp = bp
    return out


def transpose(x: Var, axes) -> Var:
    out = Var(x.d.transpose(axes), (x,))
    inv = tuple(np.argsort(axes))

    def bp():
        x.g = _acc(x.g, out.g.transpose(inv))

    out._bp = bp
    return out


def mse(pred: Var, y: np.ndarray) -> Var:
    diff = pred.d - y
    out = Var(np.mean(diff * diff), (pred,))

    def bp():
        pred.g = _acc(pred.g, out.g * 2.0 * diff / diff.size)

    out._bp = bp
    return out


class Adam:
    def __init__(self, params: list[Var], lr: float = 5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.d) for p in params]
        self.v = [np.zeros_like(p.d) for p in params]
        self.t = 0

    def step(self, grad_clip: float = 1.0) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.g is None:
                continue
            g = np.clip(p.g, -grad_clip, grad_clip)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / c1
            vh = self.v[i] / c2
            p.d -= self.lr * mh / (np.sqrt(vh) + self.eps)
            p.g = None


# ================================================================== 基类
class BaseModel:
    """统一接口。`lookback` 表示预测时需要回看多少根历史（构成序列）。"""

    name = "base"
    lookback = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def importance(self) -> np.ndarray | None:
        return None


# ================================================================== 树模型
class LGBMModel(BaseModel):
    """LightGBM。小样本上刻意压低复杂度：叶子少、叶子样本数高、强正则。"""

    name = "lightgbm"

    def __init__(self, seed: int = 42, n_estimators: int = 200):
        self.seed = seed
        self.n_estimators = n_estimators
        self._m = None
        self._imp = None

    def fit(self, X, y):
        import lightgbm as lgb

        self._m = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.1,
            random_state=self.seed,
            n_jobs=1,
            verbose=-1,
        )
        self._m.fit(X, y)
        self._imp = np.asarray(self._m.feature_importances_, dtype=float)
        return self

    def predict(self, X):
        return np.asarray(self._m.predict(X), dtype=float)

    def importance(self):
        return self._imp


class XGBModel(BaseModel):
    """XGBoost。与 LightGBM 同族但分裂方式不同，可做模型对比。"""

    name = "xgboost"

    def __init__(self, seed: int = 42, n_estimators: int = 120):
        self.seed = seed
        self.n_estimators = n_estimators
        self._m = None
        self._imp = None

    def fit(self, X, y):
        import xgboost as xgb

        self._m = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.1,
            random_state=self.seed,
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
        )
        self._m.fit(X, y)
        self._imp = np.asarray(self._m.feature_importances_, dtype=float)
        return self

    def predict(self, X):
        return np.asarray(self._m.predict(X), dtype=float)

    def importance(self):
        return self._imp


class GPRModel(BaseModel):
    """高斯过程回归。非线性、自带不确定性，但对样本量敏感 → 只取最近 N 个。

    **核超参刻意冻结**：让 sklearn 去优化 ARD 长度尺度，每折要跑几百次
    L-BFGS 迭代，400×400 的 Cholesky 重复几十遍，单折就从 10ms 涨到 1s 以上，
    30 折 walk-forward 直接卡死。而因子已经 z-score 过（量纲统一、方差为 1），
    长度尺度取 3.0 是个合理先验，冻结后精度损失很小、速度提升两个数量级。
    """

    name = "gpr"

    def __init__(self, max_train: int = 400, seed: int = 42, length_scale: float = 3.0):
        self.max_train = max_train
        self.seed = int(seed)
        self.length_scale = float(length_scale)
        self._m = None
        self._ymu = 0.0
        self._ysd = 1.0

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        if len(X) > self.max_train:
            X, y = X[-self.max_train :], y[-self.max_train :]
        self._ymu = float(np.mean(y))
        self._ysd = float(np.std(y)) or 1.0
        kernel = ConstantKernel(1.0, "fixed") * RBF(
            length_scale=np.ones(X.shape[1]) * self.length_scale,
            length_scale_bounds="fixed",
        ) + WhiteKernel(0.05, "fixed")
        self._m = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False, alpha=1e-6, n_restarts_optimizer=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._m.fit(X, (y - self._ymu) / self._ysd)
        return self

    def predict(self, X):
        return np.asarray(self._m.predict(X), dtype=float) * self._ysd + self._ymu


# ================================================================== Transformer
def _pos_encoding(length: int, dim: int) -> np.ndarray:
    """正弦位置编码。固定不训练，省参数也省得发散。"""
    pos = np.arange(length)[:, None].astype(float)
    i = np.arange(dim)[None, :].astype(float)
    ang = pos / np.power(10000.0, (2 * (i // 2)) / max(dim, 1))
    pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
    return pe[None, :, :]


class TransformerModel(BaseModel):
    """单层多头自注意力编码器（numpy 实现，见模块头部说明）。

    输入是「最近 lookback 根 K 线的因子向量」构成的序列，输出下一期收益的预测。
    多因子之间的交互由注意力学出来，位置编码让它知道先后顺序。
    """

    name = "transformer"

    def __init__(
        self,
        lookback: int = 8,
        d_model: int = 8,
        n_heads: int = 2,
        n_layers: int = 1,
        epochs: int = 30,
        lr: float = 5e-3,
        seed: int = 42,
        batch_size: int = 256,
    ):
        self.lookback = int(lookback)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self._params: list[Var] = []
        self._ymu = 0.0
        self._ysd = 1.0
        self._nf = 0

    # -------------------------------------------------- 参数初始化
    def _init(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        d, f = self.d_model, n_features
        self._nf = f
        self._params = []

        def w(*shape, gain=None):
            fan_in = shape[-2] if len(shape) >= 2 else shape[0]
            g = gain if gain is not None else np.sqrt(2.0 / max(fan_in, 1))
            v = Var(rng.normal(0.0, g, size=shape))
            self._params.append(v)
            return v

        def b(*shape):
            v = Var(np.zeros(shape))
            self._params.append(v)
            return v

        def ln():
            g, be = Var(np.ones(d)), Var(np.zeros(d))
            self._params.extend([g, be])
            return g, be

        self._Win, self._bin = w(f, d), b(d)
        self._blocks = []
        for _ in range(self.n_layers):
            self._blocks.append(
                {
                    "g1": ln(), "Wq": w(d, d), "Wk": w(d, d), "Wv": w(d, d),
                    "Wo": w(d, d), "bo": b(d),
                    "g2": ln(), "W1": w(d, 2 * d), "b1": b(2 * d),
                    "W2": w(2 * d, d), "b2": b(d),
                }
            )
        self._gout = ln()
        self._Wout, self._bout = w(d, 1), b(1)

    # -------------------------------------------------- 前向
    def _forward(self, Xseq: np.ndarray) -> Var:
        B, L, _ = Xseq.shape
        d, H = self.d_model, self.n_heads
        dh = d // H

        h = add(matmul(Var(Xseq), self._Win), self._bin)
        h = add_const(h, _pos_encoding(L, d))

        for blk in self._blocks:
            a = layernorm(h, *blk["g1"])
            q = matmul(a, blk["Wq"])
            k = matmul(a, blk["Wk"])
            v = matmul(a, blk["Wv"])
            # (B, L, d) → (B, H, L, dh)
            q = transpose(reshape(q, (B, L, H, dh)), (0, 2, 1, 3))
            k = transpose(reshape(k, (B, L, H, dh)), (0, 2, 1, 3))
            v = transpose(reshape(v, (B, L, H, dh)), (0, 2, 1, 3))
            score = scale(matmul(q, transpose(k, (0, 1, 3, 2))), 1.0 / np.sqrt(dh))
            att = matmul(softmax_last(score), v)
            att = reshape(transpose(att, (0, 2, 1, 3)), (B, L, d))
            h = add(h, add(matmul(att, blk["Wo"]), blk["bo"]))

            f = layernorm(h, *blk["g2"])
            f = relu(add(matmul(f, blk["W1"]), blk["b1"]))
            h = add(h, add(matmul(f, blk["W2"]), blk["b2"]))

        h = layernorm(h, *self._gout)
        pooled = take_last(h)
        return add(matmul(pooled, self._Wout), self._bout)

    def _seq(self, X: np.ndarray) -> np.ndarray:
        """(n, f) → (n-L+1, L, f)。用滑动窗口视图再拷贝，省内存。"""
        from numpy.lib.stride_tricks import sliding_window_view

        if len(X) < self.lookback:
            return np.empty((0, self.lookback, X.shape[1]))
        return np.ascontiguousarray(sliding_window_view(X, self.lookback, axis=0).transpose(0, 2, 1))

    # -------------------------------------------------- 训练 / 预测
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(X) <= self.lookback + 10:
            raise ValueError("样本太少，无法训练 Transformer")
        self._init(X.shape[1])
        seq = self._seq(X)
        # 第 i 个窗口的最后一个时间步对应原始第 i+L-1 行
        tgt = y[self.lookback - 1 :]
        n = min(len(seq), len(tgt))
        seq, tgt = seq[:n], tgt[:n]

        self._ymu = float(tgt.mean())
        self._ysd = float(tgt.std()) or 1.0
        ys = (tgt - self._ymu) / self._ysd

        opt = Adam(self._params, lr=self.lr)
        rng = np.random.default_rng(self.seed + 1)
        bs = min(self.batch_size, max(32, n // 2))
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, bs):
                idx = order[s : s + bs]
                pred = self._forward(seq[idx])
                loss = mse(pred, ys[idx].reshape(-1, 1))
                for p in self._params:
                    p.g = None
                backward(loss)
                opt.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(len(X), np.nan)
        if len(X) < self.lookback:
            return out
        seq = self._seq(X)
        pred = self._forward(seq)
        vals = pred.d.reshape(-1) * self._ysd + self._ymu
        out[self.lookback - 1 :] = vals
        return out

    def importance(self):
        """输入层权重的列范数，作为各因子的「进入强度」粗代理。"""
        if not self._params:
            return None
        return np.linalg.norm(self._Win.d, axis=1)


# ================================================================== 线性
class LinearModel(BaseModel):
    """岭回归。

    它存在的意义不是「更准」，而是**当基准**：树模型和神经网络的分数如果打不过
    一条岭回归，说明那点复杂度全用在拟合噪声上了。系数直接就是因子权重，
    可解释性也最好 —— 用户能一眼看出「这条策略到底在赌哪几个因子」。

    用 numpy 闭式解而不是 sklearn：一个 (k+1)×(k+1) 的正规方程，k 只有几十，
    `np.linalg.solve` 就是精确解，没有迭代、没有随机种子、没有版本差异。
    """

    name = "linear"

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._w: np.ndarray | None = None
        self._b = 0.0
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[keep], y[keep]
        if len(y) < X.shape[1] + 2:
            raise ValueError("样本太少，无法训练线性模型")

        self._mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0          # 常数列不能除零
        self._sd = sd
        Z = (X - self._mu) / sd
        Z1 = np.hstack([np.ones((len(Z), 1)), Z])

        # 截距不惩罚，否则目标整体偏移会被往 0 拉
        pen = np.eye(Z1.shape[1]) * self.alpha
        pen[0, 0] = 0.0
        coef = np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ y)
        self._b = float(coef[0])
        self._w = coef[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(len(X), np.nan)
        if self._w is None or self._mu is None:
            return out
        keep = np.isfinite(X).all(axis=1)
        if not keep.any():
            return out
        Z = (X[keep] - self._mu) / self._sd
        out[keep] = Z @ self._w + self._b
        return out

    def importance(self):
        return None if self._w is None else np.abs(self._w)


# ================================================================== 图神经网络
class GNNModel(BaseModel):
    """因子图神经网络：把每个因子当成一个节点，让节点之间互相传消息。

    一层图卷积：

        A  = softmax_rows(A_param)        # k×k，学出来的邻接矩阵
        H  = relu(X · Aᵀ · W1 + b1)       # 一行一个样本，按 A 做邻居聚合
        y  = H · w2 + b2

    `A[i][j]` 表示「因子 i 的消息有多少流向因子 j」。用行 softmax 而不是
    对称归一化（D^-1/2 A D^-1/2）：行 softmax 本身就是归一化的，不必再算度矩阵，
    而度矩阵的梯度会多出一串容易写错的项。

    ## 为什么是「因子图」而不是「品种图」

    平台的机器学习是**单品种时序**训练 —— 一次只有一条品种的样本，没有横截面，
    建不出品种之间的图。把因子当节点就没有这个问题，而且学出来的 A 本身就是
    一张「因子关联图」，正好回答「哪几个因子在互相解释」。

    ## 为什么不用 torch

    和 Transformer 一样：torch 的 CPU 轮子 1GB 起步，安装包会从 ~99MB 涨到 1GB。
    这里复用模块里的 `Var` 自动微分，几十行就够。
    """

    name = "gnn"

    def __init__(
        self,
        hidden: int = 8,
        epochs: int = 60,
        lr: float = 5e-3,
        seed: int = 42,
        batch_size: int = 256,
    ):
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self._params: list[Var] = []
        self._A: Var | None = None
        self._W1: Var | None = None
        self._b1: Var | None = None
        self._w2: Var | None = None
        self._b2: Var | None = None
        self._ymu = 0.0
        self._ysd = 1.0

    def _init(self, k: int) -> None:
        rng = np.random.default_rng(self.seed)
        h = self.hidden
        self._A = Var(rng.normal(0.0, 0.1, size=(k, k)))
        self._W1 = Var(rng.normal(0.0, np.sqrt(2.0 / max(k, 1)), size=(k, h)))
        self._b1 = Var(np.zeros(h))
        self._w2 = Var(rng.normal(0.0, np.sqrt(2.0 / max(h, 1)), size=(h, 1)))
        self._b2 = Var(np.zeros(1))
        self._params = [self._A, self._W1, self._b1, self._w2, self._b2]

    def _forward(self, X: np.ndarray) -> Var:
        # 样本是行向量，所以聚合写成 X·Aᵀ 而不是 A·X：
        # 第 j 列的新特征 = Σ_i A[j,i]·X[:,i]，即「j 收下所有邻居 i 的消息」。
        adj_t = transpose(softmax_last(self._A), (1, 0))
        h = matmul(matmul(Var(X), adj_t), self._W1)
        h = relu(add(h, self._b1))
        return add(matmul(h, self._w2), self._b2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[keep], y[keep]
        if len(y) <= X.shape[1] + 10:
            raise ValueError("样本太少，无法训练 GNN")

        self._init(X.shape[1])
        self._ymu = float(y.mean())
        self._ysd = float(y.std()) or 1.0
        ys = (y - self._ymu) / self._ysd

        opt = Adam(self._params, lr=self.lr)
        rng = np.random.default_rng(self.seed + 1)
        n = len(ys)
        bs = min(self.batch_size, max(32, n // 2))
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, bs):
                idx = order[s : s + bs]
                pred = self._forward(X[idx])
                loss = mse(pred, ys[idx].reshape(-1, 1))
                for p in self._params:
                    p.g = None
                backward(loss)
                opt.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.full(len(X), np.nan)
        if self._A is None:
            return out
        keep = np.isfinite(X).all(axis=1)
        if not keep.any():
            return out
        pred = self._forward(X[keep])
        out[keep] = pred.d.reshape(-1) * self._ysd + self._ymu
        return out

    def importance(self):
        """输入层权重的列范数，作为各因子的「进入强度」粗代理。

        和 Transformer 同一个口径，方便两种模型横向比。真正的因子关联要看
        `adjacency()`。
        """
        if self._W1 is None:
            return None
        return np.linalg.norm(self._W1.d, axis=1)

    def adjacency(self) -> np.ndarray | None:
        """学出来的因子邻接矩阵（行归一化）。第 i 行是「因子 i 把消息分给了谁」。"""
        if self._A is None:
            return None
        x = self._A.d - self._A.d.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)


# ================================================================== 注册表
MODELS: dict[str, dict] = {
    "lightgbm": {"label": "LightGBM", "cls": LGBMModel, "desc": "梯度提升树，对非线性与因子交互最敏感，速度最快。"},
    "xgboost": {"label": "XGBoost", "cls": XGBModel, "desc": "二阶梯度提升树，正则更强，与 LightGBM 互为对照。"},
    "gpr": {"label": "高斯过程 GPR", "cls": GPRModel, "desc": "核方法，小样本稳健、天然带不确定性；训练量受 O(n³) 限制。"},
    "transformer": {"label": "Transformer", "cls": TransformerModel, "desc": "单层多头自注意力，从因子序列里学时序交互（numpy 实现）。"},
    "linear": {"label": "线性回归 Ridge", "cls": LinearModel, "desc": "因子线性加权。当基准用：复杂模型打不过它，说明复杂度都用在拟合噪声上了。"},
    "gnn": {"label": "图神经网络 GNN", "cls": GNNModel, "desc": "把因子当节点学一张关联图，消息传递后再回归（numpy 实现）。"},
}

DEFAULT_MODEL = "lightgbm"


def build(key: str, seed: int = 42, **kw) -> BaseModel:
    spec = MODELS.get(key)
    if spec is None:
        raise KeyError(f"未知模型 {key}")
    return spec["cls"](seed=seed, **kw)


_PROBE = {
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "gpr": "sklearn",
    "transformer": None,  # 纯 numpy，无外部依赖
    "linear": None,
    "gnn": None,
}


def available(why: dict | None = None) -> list[str]:
    """按依赖可用性过滤：没装 lightgbm 就不该在界面上让你选。

    传 `why` 进来可以拿到每个不可用模型的失败原因。这一点是必须的：
    `import xgboost` 在 macOS 上失败时抛的是 `XGBoostError`（ctypes 加载
    dylib 失败），跟「没装这个包」完全不是一回事，但界面上看起来一样 ——
    都只是「少了这个选项」。静默吞异常会让打包问题一直藏到用户手里。
    """
    out = []
    for k in MODELS:
        mod = _PROBE.get(k)
        if mod is None:
            out.append(k)
            continue
        try:
            __import__(mod)
        except Exception as exc:
            if why is not None:
                why[k] = f"{type(exc).__name__}: {exc}"
            continue
        out.append(k)
    return out


def diagnose() -> str:
    """给自检用的一行诊断：可用模型 + 不可用的原因（含原生库路径）。"""
    why: dict = {}
    avail = available(why=why)
    parts = ["、".join(MODELS[k]["label"] for k in avail) or "无"]
    for k, msg in why.items():
        detail = msg
        if k == "xgboost":
            detail += f"（{_xgboost_lib_hint()}）"
        parts.append(f"{MODELS[k]['label']} 不可用 —— {detail}")
    return "；".join(parts)


def _xgboost_lib_hint() -> str:
    """xgboost 的原生库是运行时按路径找的，打没打进包只能这么查。"""
    try:
        import os

        import xgboost

        root = os.path.dirname(os.path.abspath(xgboost.__file__))
        lib = os.path.join(root, "lib")
        if not os.path.isdir(lib):
            return f"{lib} 不存在"
        names = sorted(os.listdir(lib))
        return f"lib 目录 {lib} → {names or '空'}"
    except Exception as exc:  # noqa: BLE001
        return f"连 xgboost 包目录都定位不到（{type(exc).__name__}: {exc}）"


def permutation_importance(
    model: BaseModel, X: np.ndarray, y: np.ndarray, n_repeat: int = 3, seed: int = 0
) -> np.ndarray | None:
    """对没有原生重要度的模型（GPR/Transformer）做置换重要度。

    在预测集上逐列打乱，看预测误差涨多少。比「权重范数」诚实得多。
    """
    if len(X) < 30:
        return None
    rng = np.random.default_rng(seed)
    base = model.predict(X)
    ok = np.isfinite(base)
    if ok.sum() < 30:
        return None
    base_err = float(np.mean((base[ok] - y[ok]) ** 2))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        errs = []
        for _ in range(n_repeat):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            p = model.predict(Xp)
            m = np.isfinite(p)
            if m.sum() < 30:
                continue
            errs.append(float(np.mean((p[m] - y[m]) ** 2)))
        imp[j] = max(0.0, (np.mean(errs) if errs else base_err) - base_err)
    return imp
