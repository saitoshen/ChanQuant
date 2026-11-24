# -*- coding: utf-8 -*-
"""核对 `LGBMModel` 两套参数：大折必须是原项目逐字，小折必须能分裂。"""
import sys

import numpy as np

sys.path.insert(0, "D:/ChanQuant-Dev")
from ttrade import mlmodels as ML  # noqa: E402

rng = np.random.default_rng(7)


def show(tag, n, sd=0.01):
    X = rng.normal(size=(n, 20))
    y = X[:, 0] * sd * 0.3 + rng.normal(scale=sd, size=n)
    m = ML.build("lightgbm")
    m.fit(X, y)
    inner = m._m
    pr = m.predict(X)
    print(f"\n== {tag}：{n} 行 / 标签 sd={sd}")
    print("  内部估计器 :", type(inner).__module__ + "." + type(inner).__name__)
    print("  树数       :", inner.n_estimators)
    print("  实际超参   :", {k: inner.get_params()[k] for k in (
        "learning_rate", "num_leaves", "min_child_samples", "max_depth",
        "colsample_bytree", "subsample", "subsample_freq",
        "reg_alpha", "reg_lambda")})
    print("  预测取值数 :", len(np.unique(np.round(pr, 12))))
    print("  叶子总数   :",
          sum(t["num_leaves"] for t in inner.booster_.dump_model()["tree_info"]))


show("大折（原项目参数）", 20000)
show("大折·标签量纲很小时也不许退化成常数", 20000, sd=0.004)
show("小折（保护分支，自检用的 54 行）", 54, sd=0.005)
show("小折 399 行", 399, sd=0.005)
