# MOYO OOD Evaluation Draft

This note organizes the current MOYO-based OOD results into paper-ready text.

Related assets:

- Overview figure: [render/moyo_overview/first_frame_grid.png](/home/hanlong/project/mmlphuman/render/moyo_overview/first_frame_grid.png)
- Processed motion windows: [amass/moyo_eval_240/manifest.json](/home/hanlong/project/mmlphuman/amass/moyo_eval_240/manifest.json)
- OOD severity statistics: [amass/moyo_eval_240/ood_stats.json](/home/hanlong/project/mmlphuman/amass/moyo_eval_240/ood_stats.json)
- `w/ PCA` renders: [render/moyo_pca](/home/hanlong/project/mmlphuman/render/moyo_pca)
- `w/o PCA` renders: [render/moyo_nopca](/home/hanlong/project/mmlphuman/render/moyo_nopca)

## Paper Subsection

### English version

**Strong OOD Evaluation on MOYO.** We further evaluate the robustness of the trained avatar under strongly out-of-distribution motions using the SMPL-X neutral release of MOYO. We select four challenging yoga sequences, namely *Scorpion*, *Cockerel*, *Rajakapotasana*, and *Split*, and extract one 240-frame high-motion window from each sequence. Since MOYO does not provide paired ground-truth images for our target subject, this experiment is used as a **qualitative stress test** rather than a full-reference benchmark.

To quantify how far these motions deviate from the training distribution, we fit a 20-dimensional PCA space on the body poses observed during training and measure the per-frame pose coefficients in units of standard deviation. The selected MOYO sequences are substantially outside the training pose manifold, with mean absolute z-scores ranging from 2.03 to 4.63 and maximum deviations reaching 27.44. Moreover, 90% to 100% of the frames exceed the 2-sigma range in at least one PCA component, confirming that this setting represents a strong pose OOD regime.

We then compare test-time rendering **with** and **without** the PCA-based pose regularization used in the original method. Without PCA, extreme MOYO poses frequently lead to structural artifacts, including radial spikes, ghosting, and local blur, especially on *Cockerel*, *Rajakapotasana*, and *Scorpion*. In contrast, enabling PCA produces substantially cleaner and more stable renderings while preserving the overall identity and garment appearance. These results suggest that the PCA prior is particularly helpful under strong pose shift, where raw pose extrapolation becomes unstable.

### Chinese reference version

**MOYO 强 OOD 动作评估。** 我们进一步使用 MOYO 的 SMPL-X neutral 动作数据评估训练后 avatar 在强分布外动作下的鲁棒性。具体地，我们选取了四段高难瑜伽动作，分别为 *Scorpion*、*Cockerel*、*Rajakapotasana* 和 *Split*，并从每段动作中截取一个 240 帧的高运动窗口。由于 MOYO 不提供与目标人物对应的真实图像，这一实验更适合作为**强 OOD 的定性压力测试**，而不是有真值的重建指标评测。

为了量化这些动作偏离训练分布的程度，我们先对训练阶段出现过的 body pose 拟合 20 维 PCA 空间，再将 MOYO 动作投影到该空间，并以标准差单位统计其偏离程度。结果表明，这四段动作都显著超出了训练姿态流形：平均绝对 z-score 介于 2.03 到 4.63 之间，最大偏离达到 27.44，并且 90% 到 100% 的帧在至少一个 PCA 分量上超过 2-sigma 范围，说明这是一个强 pose OOD 设定。

在此基础上，我们比较了**开启**与**关闭**测试时 PCA 正则的渲染效果。关闭 PCA 时，极端动作容易引发明显的结构性伪影，包括从人体向外发散的尖刺、局部 ghosting 以及模糊，尤其在 *Cockerel*、*Rajakapotasana* 和 *Scorpion* 上更明显。相比之下，开启 PCA 后渲染结果明显更稳定、更干净，同时仍能保持整体身份和服装外观。这说明该 PCA 先验在强姿态偏移下能有效缓解直接外推带来的不稳定问题。

## Figure Caption

### Main-paper caption

**Qualitative comparison on strong OOD motions from MOYO.** We render four challenging yoga motions from MOYO (*Scorpion*, *Cockerel*, *Rajakapotasana*, and *Split*), which lie far outside the training pose manifold. Left: rendering with test-time PCA pose regularization. Right: rendering without PCA. Disabling PCA leads to severe artifacts such as radial spikes, ghosting, and local blur, whereas PCA regularization yields noticeably cleaner and more stable renderings under extreme pose shift.

### Short caption

**MOYO strong OOD stress test.** Test-time PCA regularization substantially improves rendering stability under extreme unseen motions, reducing spike-like artifacts and ghosting.

## Table Text

### Table caption

**OOD severity of selected MOYO motion windows.** We report the distance of each sequence from the training pose manifold using a 20D PCA space fitted on training body poses. `Frames > 2σ` denotes the fraction of frames for which at least one PCA component falls outside the 2-sigma range.

### LaTeX table

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
Sequence & Frames & Mean $|z|$ & Max $|z|$ & Frames $> 2\sigma$ \\
\midrule
Scorpion        & 240 & 2.38 & 15.52 & 0.90 \\
Cockerel        & 240 & 4.54 & 22.30 & 1.00 \\
Rajakapotasana  & 240 & 4.63 & 27.44 & 1.00 \\
Split           & 240 & 2.03 & 9.64  & 1.00 \\
\bottomrule
\end{tabular}
\caption{OOD severity of the selected MOYO motion windows measured in a 20D PCA space fitted on training body poses. Larger values indicate stronger deviation from the training pose manifold.}
\label{tab:moyo_ood}
\end{table}
```

### Optional qualitative table note

You can add the following sentence below the table if you want a stronger connection to the figure:

> Among the selected motions, *Cockerel* and *Rajakapotasana* exhibit the largest pose deviation and also produce the most severe rendering artifacts when PCA regularization is disabled.

## Suggested Paper Positioning

- Use this MOYO experiment as a **strong OOD qualitative stress test**, not as the only main quantitative benchmark.
- Keep the main quantitative tables on settings with ground truth, such as `Pose OOD, View ID` or camera-held-out `View OOD`.
- Use MOYO to support the claim that PCA helps under extreme pose shift where direct extrapolation becomes visually unstable.
