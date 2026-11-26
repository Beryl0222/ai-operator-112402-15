// 分组权重下的无偏两样本核距离，以及双档误差合同下的置信区间。
//
// 估计量（同组对全部剔除，同侧有序对、跨侧共享组剔除）：
//   D = K_A/D_A + K_B/D_B - 2 K_AB/D_AB
// 其中 K 为草图核和，D 为对应权重对数。草图核与精确核的逐元素偏差不超过
// 该档 absoluteError，归一化均值误差按 ε·(有效权重对数和)/(权重对数) 传播。

import {TIERS} from './kernel.mjs';
import {crossKernelNumerator, crossWeightDenominator} from './groups.mjs';

function sideTerm(store, tier, {groupSet = null} = {}) {
  const reducer = tier === TIERS.P2 ? store.p2 : store.v1;
  const denom = reducer.weightDenominator(groupSet);
  const numer = reducer.kernelNumerator(groupSet);
  return {
    numerator: numer,
    denominator: denom,
    ready: denom > 0,
    mean: denom > 0 ? numer / denom : null,
  };
}

function crossTerm(storeA, storeB, tier, {groupSet = null} = {}) {
  const {numerator} = crossKernelNumerator(storeA, storeB, tier, groupSet);
  const denominator = crossWeightDenominator(storeA, storeB, tier, groupSet);
  return {
    numerator,
    denominator,
    ready: denominator > 0,
    mean: denominator > 0 ? numerator / denominator : null,
  };
}

/**
 * 主入口：计算距离点估计与误差区间。
 *
 * @param {object} spec
 * @param {import('./groups.mjs').SideStore} spec.storeA 旧麦侧
 * @param {import('./groups.mjs').SideStore} spec.storeB 新麦侧
 * @param {{v1:number, p2:number}} spec.errors 两档统一最大核误差
 * @param {boolean} [spec.useDiagnostics] 是否消费已升档的 precision2 组
 * @returns {object} 点估计、区间、每侧就绪状态、分层误差与不可比标记
 */
export function groupedDistance({storeA, storeB, errors, useDiagnostics = true}) {
  const comparable = storeA.projection === storeB.projection;
  const v1 = computeTier(storeA, storeB, TIERS.V1, errors.v1, null);

  const upgradedA = new Set(storeA.upgradedGroups());
  const upgradedB = new Set(storeB.upgradedGroups());
  const p2GroupSet = new Set([...upgradedA, ...upgradedB]);
  const diagnosticsAvailable = useDiagnostics && p2GroupSet.size > 0;
  const p2 = diagnosticsAvailable
    ? computeTier(storeA, storeB, TIERS.P2, errors.p2, p2GroupSet)
    : null;

  // 混合档：已双侧升档的对享受 p2 误差，其余对仍只有 small-v1 误差证据。
  // p2 数据存在但组太少（任一项无有效对）时，该子问题证不出，回退 v1 口径。
  let merged;
  if (p2 && p2.ready) {
    merged = mergeTiers(storeA, storeB, v1, p2, p2GroupSet, errors);
  } else {
    merged = {...v1, diagnosticNote: p2 ? 'p2 升档组不足以形成有效跨组对，子问题回退 v1' : null};
  }

  return {
    comparable,
    projection: {a: storeA.projection, b: storeB.projection},
    point: merged.point,
    bound: merged.bound,
    interval: merged.ready ? [merged.point - merged.bound, merged.point + merged.bound] : null,
    ready: merged.ready,
    insufficientSides: merged.insufficientSides,
    tiers: {v1, p2, merged},
    diagnosticGroups: [...p2GroupSet].sort((a, b) => a - b),
  };
}

function computeTier(storeA, storeB, tier, epsilon, groupSet) {
  const a = sideTerm(storeA, tier, {groupSet});
  const b = sideTerm(storeB, tier, {groupSet});
  const ab = crossTerm(storeA, storeB, tier, {groupSet});
  const ready = a.ready && b.ready && ab.ready;
  const point = ready ? a.mean + b.mean - 2 * ab.mean : null;
  // 每个归一化核均值的误差不超过 epsilon（分子每项偏差 epsilon，
  // 权重对数和恰好等于分母）。三项线性组合，区间半宽 = 4 epsilon。
  const bound = ready ? 4 * epsilon : null;
  return {
    tier,
    epsilon,
    ready,
    point,
    bound,
    interval: ready ? [point - bound, point + bound] : null,
    terms: {a, b, ab},
    insufficientSides: [
      ...(!a.ready ? ['a'] : []),
      ...(!b.ready ? ['b'] : []),
      ...(!ab.ready ? ['ab'] : []),
    ],
  };
}

/**
 * 混合档合并：在 v1 全量估计上，用 p2 子空间结果替换升档组真正影响的子问题。
 * 未升档组涉及的对保持 v1 误差；只要一侧的核对中有未升档组，该对依旧只有
 * small-v1 的误差证据。
 */
function mergeTiers(storeA, storeB, v1, p2, p2GroupSet, errors) {
  if (!v1.ready) {
    return {...v1, point: null, bound: null, interval: null, ready: false};
  }
  // 点估计：p2 覆盖的组用二阶草图重算，未覆盖组沿用 v1 草图。
  // 两侧未升档组之间以及“升档/未升档”之间的对仍取 v1 近似，
  // 做法是 v1 全量点估计减去 v1 子集项再加上 p2 子集项。
  const v1Sub = computeTier(storeA, storeB, TIERS.V1, errors.v1, p2GroupSet);
  let point = v1.point;
  if (v1Sub.ready && p2.ready) point = v1.point - v1Sub.point + p2.point;

  // 区间半宽逐对分层：p2 子集内的对按 4*eps_p2，其余对按 4*eps_v1。
  // 用权重占比把子集贡献折算进半宽（同阶归一化均值的线性传播）。
  const share = subShare(v1, v1Sub);
  const bound = 4 * (errors.p2 * share + errors.v1 * (1 - share));
  const ready = true;
  return {
    tier: 'mixed',
    epsilon: null,
    ready,
    point,
    bound,
    interval: [point - bound, point + bound],
    terms: v1.terms,
    subShare: share,
    insufficientSides: [],
  };
}

/** p2 子集在三项归一化均值中的有效权重占比（0..1），用于误差分层。 */
function subShare(v1Full, v1Sub) {
  if (!v1Sub.ready) return 0;
  const ratios = [];
  for (const key of ['a', 'b', 'ab']) {
    const full = v1Full.terms[key].denominator;
    const sub = v1Sub.terms[key].denominator;
    if (full > 0) ratios.push(Math.min(1, sub / full));
  }
  if (ratios.length === 0) return 0;
  return ratios.reduce((x, y) => x + y, 0) / ratios.length;
}

function projectionPair(p) {
  if (p === null || p === undefined) return undefined;
  if (typeof p === 'string') return `${p}|${p}`;
  return `${p.a}|${p.b ?? p.a}`;
}

/**
 * 距离相减的漂移守卫：两个两样本距离要相减，两侧投影身份都必须一致。
 * 任一侧换了投影基且缺少公共坐标即拒绝，不允许把距离差解释为阈值漂移
 * （维度相等也不建立共同空间）。
 */
export class ProjectionMismatchError extends Error {
  constructor(projectionA, projectionB) {
    super(
      `投影身份不可比：${JSON.stringify(projectionA)} 与 ${JSON.stringify(projectionB)} ` +
        '缺少公共可核验坐标，不能把两个距离相减后解释为阈值漂移',
    );
    this.name = 'ProjectionMismatchError';
  }
}

export function assertComparable(resultA, resultB) {
  const pa = resultA.projection;
  const pb = resultB.projection;
  if (projectionPair(pa) !== projectionPair(pb)) {
    throw new ProjectionMismatchError(pa, pb);
  }
}

/** 仅在两侧投影都可比时允许计算阈值漂移量；否则抛错。 */
export function thresholdDrift(resultBefore, resultAfter) {
  assertComparable(resultBefore, resultAfter);
  return resultAfter.point - resultBefore.point;
}
