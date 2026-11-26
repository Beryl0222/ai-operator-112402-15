# 手机语音唤醒的核归约与阈值交接

这里的核码本、录音特征和观测权重是用于设备联调的合成资料。使用 Node.js 22。运行 `npm ci --offline` 核对依赖锁，`npm run inspect` 查看资料字段，`npm run scenario` 本地跑完整阈值交接脚本，`npm test` 检查资料格式、算子契约与联调痕迹。检视脚本只读取元数据。全部计算在设备本地完成，音频与特征不上传。

## 算子结构（`lib/`）

- `kernel.mjs`：精确 RBF 核及其解析梯度、两档设备草图（small-v1 / diagnostic-v1）、Taylor 特征映射的雅可比与合同导数。跨档点积直接抛 `TypeError`，不能把两种向量长度的内积当同一种核。
- `groups.mjs`：分组流式归约。按 `physicalGroup` 维护权重和与草图加权和，核和用 `总量平方 − 组内平方` 在线得到，**不常驻 N×N 核矩阵**；同组帧（同一次播放、回声分离前后）从分子分母整体剔除。`Budget` 同时记 `budgetBytes` 暂存字节与 `pairLimit` 精确对数，单 query 伴随精确求值走独立 `queryPairs` 口径。
- `distance.mjs`：分组权重下的无偏两样本核距离 `D = K_A/D_A + K_B/D_B − 2K_AB/D_AB`（同侧有序对、跨侧共享组剔除），双档误差合同下的区间，以及投影漂移守卫 `thresholdDrift`。
- `witness.mjs`：单 query 核见证（query 所在组在该侧排除）、对每个单组去留的敏感度、精确 RBF 梯度（`exact-rbf-analytic`）与草图合同梯度（`exact-taylor-v1/v2`，仅限误差合同声明路径）。
- `handoff.mjs`：门限判决 `decide`、诊断需求 `diagnosticNeeds`、追加式本地账本 `DecisionLedger`、校准会话 `CalibrationSession`（录入→距离→判决→升档→再判决→账本）。
- `scenario.mjs`：按 `actions` 顺序驱动联调，输出每步可审计痕迹与预算结账。

## 判决与误差合同

- 只有区间上界 `≤ accept` 才接受新配置，下界 `≥ reject` 才作废；落入临界区一律 `hold` 保留原阈值。
- v1 草图逐核误差 `absoluteError=0.016`，归一化均值线性传播后距离半宽为 `4ε`；precision2（`0.00081`）只对设备供应范围内（组 0–39）且预算容纳的相关组升档。只要一侧的核对中有未升档组，该对仍只有 small-v1 的误差证据；混合档半宽按升档子集的权重占比分层。
- 仍证不出时 `hold`，账本的 `diagnosticGap` 公布还缺哪侧哪组、哪些受 `budgetBytes` 阻挡、哪些设备根本不供应。
- 小样本或权重全落在一组时对应分母为 0，距离 `ready=false` 并在 `insufficientSides` 标明 `a`/`b`/`ab` 哪边无法得出。
- 投影身份不同（如仅交付 scaledV2、缺共同可核验坐标）时 `comparable=false`；维度相等不建立共同空间，`thresholdDrift` 抛 `ProjectionMismatchError`，禁止把两个距离相减解释成阈值漂移。

## 数据格式

默认核是 exp(−||x−y||²/(2σ²))。同侧分子为不同 physicalGroup 之间 w_i*w_j*k 的有序和，分母为同一集合的 w_i*w_j 和，跨侧同组也排除并按自身分母。不参与的组无梯度。见证读数对单 query 给两侧按权重的核均值之差，若 query 身份在该侧则该组从分子分母排除。最终接受策略为距离上界 ≤accept 或下界 ≥reject，其余等待核证据。可用 approximate 列给离线设备的特征映射和每个 key 的统一最大核误差。近似的梯度仅可用于派生误差合同声明的路径，报使用了何种导数。budgetBytes 包含暂存数据；pairLimit 计典型读取中真实精确成对核求值数。 small-v1 是预装设备模型，kernelCode 提供它的固定的离散特征域，kernelSketch 提供设备发布的近似特征。训练态的实数输入仍需按精确核引用梯度，不能把编码近似外推到表外。不能从核距离单独推定阈值可携带，迁移只在两边用可比较投影且录音上限下界满足门限时允许。 第二代录音可能仅交付 scaledV2 视图而缺共同的可核验投影身份。接收仅标明维度相等不能建立共同空间。 梯度请求对应指定 query 的核见证与明确的训练特征输入，pairLimit 以一轮设备阈值决定所用的精确核对计，不把单 query 伴随算入前一决定。 设备对组 0 至 39 可供应 precision2 域中的 kernelSketch(code,2)。只要一侧的核对中有未升档组，该对依旧只有 small-v1 的误差证据。可在正常预算内为确实可从源端读回的查询使用精确核，不允许把跨两种向量长度的点积当诊断档使用。

## 取数

`inputs/microphone-samples.json` 保存标定、规格和调用记录。记录中的范围除明确说明外从零开始，使用半开区间。批量数组按调用记录列出的顺序交付。kernelRow(side,index) 返回录音特征与物理组，kernelCode(code) 和 kernelSketch(code) 是设备公开的输入码本及近似特征。
