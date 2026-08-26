# -*- coding: utf-8 -*-
"""历史数据时序分布刻画（阶段1 EDA）——分析真实 4 年小时级数据的分布与时序相关性。"""
import scipy.io as sio
import numpy as np

m = sio.loadmat('history/IEEE33_PV&Load_Data.mat')
L = m['load_hourly_normalized']      # (35064, 32)
PV = m['PV_hourly_normalized']       # (35064, 4)
days = L.shape[0] // 24
nl = L.reshape(days, 24, 32)         # (days, hour, load)
nv = PV.reshape(days, 24, 4)         # (days, hour, pv)


def acf(x, maxlag):
    x = x - np.mean(x)
    v = np.var(x)
    return [(np.roll(x, -k)[:len(x) - k] * x[:len(x) - k]).mean() / v
            for k in range(1, maxlag + 1)]


print("=== 1. 时段分布（跨1461天）===")
print("  hour   load_mean  load_std   pv_mean  pv_std")
for h in range(24):
    lh = nl[:, h, :].mean(axis=1)
    print("  %2d     %.4f    %.4f   %.4f  %.4f"
          % (h, nl[:, h, :].mean(), nl[:, h, :].std(),
             nv[:, h, :].mean(), nv[:, h, :].std()))

print("\n=== 2. 时段间相关性（同一代表负荷 Load24, 420kW）===")
sel = nl[:, :, 23].reshape(-1)
a = acf(sel, 24)
print("  小时级自相关: lag1=%.3f lag2=%.3f lag24(前一日同期)=%.3f" % (a[0], a[1], a[23]))

print("\n=== 3. 变差分解（日内轨迹 vs 日间电平）===")
daylev = nl.mean(axis=1)             # (days,32) 每日平均电平
track = nl - daylev[:, None, :]      # 日内轨迹
print("  负荷: 日间电平=%.1f%%  日内轨迹=%.1f%%" % (100 * daylev.var() / nl.var(), 100 * track.var() / nl.var()))
pvlev = nv.mean(axis=1)
pvtr = nv - pvlev[:, None, :]
print("  PV:   日间电平=%.1f%%  日内轨迹=%.1f%%" % (100 * pvlev.var() / nv.var(), 100 * pvtr.var() / nv.var()))

print("\n=== 4. 日间电平自相关 / 季节 / 年际 ===")
lv = daylev.mean(axis=1)             # (days,)
al = acf(lv, 30)
print("  日间电平自相关: lag1=%.3f lag7=%.3f lag30=%.3f" % (al[0], al[6], al[29]))
print("  日电平 前365天均值=%.4f 后365天=%.4f" % (lv[:365].mean(), lv[-365:].mean()))
qdiv = np.array_split(lv, 4)         # 按年分段
print("  分年日电平均值:", ["%.4f" % s.mean() for s in qdiv])

print("\n=== 5. 共同因子（各负荷是否同涨同落）===")
daymean = nl.mean(axis=2)            # 每时刻全负荷均值 (days,24)
corr = [np.corrcoef(nl[:, :, j].reshape(-1), daymean.reshape(-1))[0, 1] for j in range(32)]
print("  各负荷与时刻全负荷均值相关: min=%.3f median=%.3f max=%.3f" % (np.min(corr), np.median(corr), np.max(corr)))

print("\n=== 6. 分布形状（峰时17 vs 谷时3）===")
for h, tag in (3, "谷(3时)"), (17, "峰(17时)"):
    v = nl[:, h, :].mean(axis=1)     # 当日该时刻 32 负荷均值
    print("  %s: mean=%.4f std=%.4f p5=%.4f p95=%.4f skew=%.3f kurt=%.3f"
          % (tag, v.mean(), v.std(), np.percentile(v, 5), np.percentile(v, 95),
             float(np.mean(((v - v.mean())/v.std())**3)),
             float(np.mean(((v - v.mean())/v.std())**4) - 3)))