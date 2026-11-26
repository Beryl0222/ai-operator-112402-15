// 核见证：对单条 query 给出两侧按权重的核均值之差（MMD 的单样本见证函数）。
// query 身份所在组在该侧从分子分母排除（leave-one-group-out），并给出对每个
// 单组去留的敏感度。梯度分两条路径：精确 RBF 解析梯度（训练态实数输入）与
// 误差合同声明路径内的 Taylor 草图合同导数。

import {
  rbf,
  rbfGradient,
  sketchDot,
  featureMapWithJacobian,
  TIERS,
} from './kernel.mjs';

function sideMean(store, tier, queryVector, excludedGroup) {
  const reducer = tier === TIERS.P2 ? store.p2 : store.v1;
  const {W, S} = reducer.aggregateWithout(excludedGroup);
  if (W <= 0) return {mean: null, W, numerator: null};
  const numerator = sketchDot(S, queryVector, reducer.tier);
  return {mean: numerator / W, W, numerator};
}

/**
 * 单 query 见证。
 * @param {object} p
 * @param {import('./groups.mjs').SideStore} p.storeA
 * @param {import('./groups.mjs').SideStore} p.storeB
 * @param {number} p.queryCode query 的码本 code（用于取草图）
 * @param {number} p.queryGroup query 的物理组；该组在对应侧排除
 * @param {number} p.side query 归属侧：0=A，1=B，-1=不属于任何一侧
 * @param {1|2} [p.precision] 草图档；2 仅当 query 与两侧统计都具备 p2 时可用
 * @param {import('./groups.mjs').Budget} [p.budget] 传入则单 query 伴随精确核对单独记账
 */
export function kernelWitness({
  storeA,
  storeB,
  queryCode,
  queryGroup,
  side,
  precision = 1,
  budget = null,
  sketchLoader,
}) {
  const tier = precision === 2 ? TIERS.P2 : TIERS.V1;
  const q = sketchLoader(queryCode, precision);
  const rank = q.length;
  const redA = tier === TIERS.P2 ? storeA.p2 : storeA.v1;
  const redB = tier === TIERS.P2 ? storeB.p2 : storeB.v1;
  if (redA.rank !== rank || redB.rank !== rank || (precision === 2 && (redA.groups.size === 0 || redB.groups.size === 0))) {
    const missing = [];
    if (redA.rank !== rank || (precision === 2 && redA.groups.size === 0)) missing.push('a');
    if (redB.rank !== rank || (precision === 2 && redB.groups.size === 0)) missing.push('b');
    throw new RangeError(`诊断档见证需要两侧相关组都已升档；缺失侧：[${missing.join(',')}]`);
  }
  const excludeA = side === 0 ? queryGroup : -1;
  const excludeB = side === 1 ? queryGroup : -1;
  const a = sideMean(storeA, tier, q, excludeA);
  const b = sideMean(storeB, tier, q, excludeB);
  if (a.mean === null || b.mean === null) {
    return {
      ready: false,
      witness: null,
      missing: [...(a.mean === null ? ['a'] : []), ...(b.mean === null ? ['b'] : [])],
      tier,
    };
  }
  const witness = a.mean - b.mean;

  // 对单组去留敏感：逐组剔除后重算见证（只动该组的累加器子问题）。
  const sensitivity = leaveOneGroupOut(storeA, storeB, tier, q, side, queryGroup);
  // 草图路径不产生精确成对求值，决定轮 pairLimit 与 queryPairs 都不增加。
  return {
    ready: true,
    witness,
    tier,
    derivative: null,
    sides: {a, b},
    sensitivity,
  };
}

function leaveOneGroupOut(storeA, storeB, tier, q, querySide, queryGroup) {
  const out = [];
  const reducers = [
    [0, tier === TIERS.P2 ? storeA.p2 : storeA.v1],
    [1, tier === TIERS.P2 ? storeB.p2 : storeB.v1],
  ];
  for (const [sideIdx, reducer] of reducers) {
    for (const group of reducer.positiveGroups()) {
      // query 自身组在本侧本来就被排除；它在对侧的去留仍有信息量，照常给出。
      const excludeA = sideIdx === 0 ? group : querySide === 0 ? queryGroup : -1;
      const excludeB = sideIdx === 1 ? group : querySide === 1 ? queryGroup : -1;
      const a = sideMean(storeA, tier, q, excludeA);
      const b = sideMean(storeB, tier, q, excludeB);
      if (a.mean === null || b.mean === null) {
        out.push({side: sideIdx, group, witness: null, influence: null, reason: '剔除后无有效权重'});
        continue;
      }
      const w = a.mean - b.mean;
      out.push({side: sideIdx, group, witness: w, influence: w});
    }
  }
  return out;
}

/**
 * 精确核见证及其对明确训练特征输入的梯度。
 * 只接受同投影空间的实数向量对；每次调用按真实精确成对核求值数记账，
 * 单 query 伴随求值计入 queryPairs，不混入阈值决定轮的 pairLimit。
 *
 * @param {object} p
 * @param {Array<{group:number, weight:number, values:Float64Array}>} p.framesA
 * @param {Array<{group:number, weight:number, values:Float64Array}>} p.framesB
 * @param {Float64Array} p.query 训练态 query 实数向量
 * @param {number} p.queryGroup
 * @param {number} p.side
 * @param {number} p.sigma
 * @param {import('./groups.mjs').Budget} [p.budget]
 * @param {Float64Array} [p.trainable] 要求梯度的训练输入；缺省取 query
 */
export function exactWitnessGradient({
  framesA,
  framesB,
  query,
  queryGroup,
  side,
  sigma,
  budget = null,
  trainable = null,
}) {
  const target = trainable ?? query;
  const grad = new Float64Array(target.length);

  const scan = (frames, excluded) => {
    let W = 0;
    let N = 0;
    let sum = 0;
    const gradSum = new Float64Array(target.length);
    for (const f of frames) {
      if (f.weight === 0) continue;
      if (excluded && f.group === queryGroup) continue;
      W += f.weight;
      N += 1;
      sum += f.weight * rbf(query, f.values, sigma);
      if (target === query) {
        const gk = rbfGradient(query, f.values, sigma);
        for (let d = 0; d < gradSum.length; d++) gradSum[d] += f.weight * gk[d];
      } else if (f.values === target) {
        // 训练输入是某侧帧向量：只有与它对应的核项贡献梯度
        const gk = rbfGradient(target, query, sigma);
        for (let d = 0; d < gradSum.length; d++) gradSum[d] += f.weight * gk[d];
      }
    }
    if (budget) budget.chargeExact(N, {query: true});
    return {W, N, sum, gradSum};
  };

  const A = scan(framesA, side === 0);
  const B = scan(framesB, side === 1);
  if (A.W <= 0 || B.W <= 0) {
    return {ready: false, witness: null, gradient: null, derivative: 'exact-rbf-analytic'};
  }
  const witness = A.sum / A.W - B.sum / B.W;

  if (target === query) {
    for (let d = 0; d < grad.length; d++) grad[d] = A.gradSum[d] / A.W - B.gradSum[d] / B.W;
  } else {
    for (let d = 0; d < grad.length; d++) grad[d] = A.gradSum[d] / A.W - B.gradSum[d] / B.W;
  }

  return {
    ready: true,
    witness,
    gradient: grad,
    derivative: 'exact-rbf-analytic',
    tier: 'exact',
    exactPairs: A.N + B.N,
  };
}

/**
 * 草图合同路径的见证梯度：映射 phi(query) 的雅可比乘以两侧加权草图均值差。
 * 仅可用于派生误差合同声明的路径，返回值显式标注 derivative 与档位。
 */
export function sketchWitnessGradient({
  storeA,
  storeB,
  queryValues,
  queryGroup,
  side,
  sigma,
  precision = 1,
}) {
  const tier = precision === 2 ? TIERS.P2 : TIERS.V1;
  const {phi, jac, derivative} = featureMapWithJacobian(queryValues, sigma, precision);
  const reducerA = tier === TIERS.P2 ? storeA.p2 : storeA.v1;
  const reducerB = tier === TIERS.P2 ? storeB.p2 : storeB.v1;
  const aggA = reducerA.aggregateWithout(side === 0 ? queryGroup : -1);
  const aggB = reducerB.aggregateWithout(side === 1 ? queryGroup : -1);
  if (aggA.W <= 0 || aggB.W <= 0) {
    return {ready: false, gradient: null, derivative, tier};
  }
  const meanA = Float64Array.from(aggA.S, (v) => v / aggA.W);
  const meanB = Float64Array.from(aggB.S, (v) => v / aggB.W);
  const grad = new Float64Array(queryValues.length);
  for (let r = 0; r < phi.length; r++) {
    const coef = meanA[r] - meanB[r];
    for (let d = 0; d < grad.length; d++) grad[d] += coef * jac[r][d];
  }
  return {ready: true, gradient: grad, derivative, tier};
}
