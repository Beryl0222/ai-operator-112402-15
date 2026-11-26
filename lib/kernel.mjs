// 核求值、特征草图与误差合同。
//
// 两档特征各自张成各自的核近似空间，禁止跨档做点积（长度不同也无法做）。
// v1 草图对应一阶 Taylor 展开，precision2 对应二阶展开；误差上限由设备
// 在 inputs 里以 absoluteError 给出，本模块只负责按合同记账与求导路径标记。

import {kernelCode, kernelSketch} from '../inputs/microphone-samples.mjs';

export const TIERS = Object.freeze({V1: 'small-v1', P2: 'diagnostic-v1'});

/** 精确 RBF 核 k(x,y)=exp(-||x-y||^2/(2 sigma^2))。 */
export function rbf(x, y, sigma) {
  let s = 0;
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) {
    const d = x[i] - y[i];
    s += d * d;
  }
  return Math.exp(-s / (2 * sigma * sigma));
}

/**
 * 精确核的解析梯度 dk/dx。
 * 训练态实数输入只能走这条引用梯度（README 契约），不可把编码近似外推到表外。
 */
export function rbfGradient(x, y, sigma) {
  if (x.length !== y.length) {
    throw new RangeError('精确核梯度要求两侧向量处于同一可核验投影空间');
  }
  const k = rbf(x, y, sigma);
  const c = k / (sigma * sigma);
  return Float64Array.from(x, (xi, i) => c * (y[i] - xi));
}

/**
 * 取设备发布的草图特征。precision=1 为 small-v1，precision=2 为 diagnostic-v1。
 * 草图仅对码本内 code 定义；scaledV2 等无共同投影身份的视图没有对应草图。
 */
export function sketchFor(code, precision) {
  return kernelSketch(code, precision);
}

/** 同档两张草图的点积，即该档对精确核的近似。跨档直接拒绝。 */
export function sketchDot(u, v, tierU, tierV = tierU) {
  if (tierU !== tierV || u.length !== v.length) {
    throw new TypeError(
      `禁止把两档特征的内积当作同一种核：${tierU}(${u.length}) x ${tierV}(${v.length})`,
    );
  }
  let s = 0;
  for (let i = 0; i < u.length; i++) s += u[i] * v[i];
  return s;
}

/**
 * 特征映射 phi_t(x) 及其对训练输入 x 的解析雅可比。
 *
 * derivative 标记导数来源：
 *  - 'exact-taylor-v1'：一阶 Taylor 映射 exp(-q/2)*[1, x/sigma] 的解析导数，
 *    只允许用于派生 small-v1 误差合同声明的路径；
 *  - 'exact-taylor-v2'：二阶 Taylor 映射（含 x_i x_j 交叉项）的解析导数，
 *    只允许用于派生 diagnostic-v1 误差合同声明的路径。
 *
 * 草图本身由 code 生成；训练态调用方持有真实向量 x 时用本函数拿到
 * 与草图同构的映射及雅可比，不把编码近似外推到码本表外。
 */
export function featureMapWithJacobian(x, sigma, precision) {
  const d = x.length;
  const inv = 1 / sigma;
  const q = x.reduce((a, xi) => a + xi * xi, 0) * inv * inv;
  const envelope = Math.exp(-q / 2);
  const terms = []; // {coef, indices}，value = envelope*coef*prod(x_i)
  terms.push({coef: 1, indices: []});
  for (let i = 0; i < d; i++) terms.push({coef: inv, indices: [i]});
  if (precision === 2) {
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        terms.push({coef: inv * inv / (i === j ? Math.SQRT2 : 1), indices: [i, j]});
      }
    }
  }
  const m = terms.length;
  const phi = new Float64Array(m);
  const jac = Array.from({length: m}, () => new Float64Array(d));
  for (let r = 0; r < m; r++) {
    const {coef, indices} = terms[r];
    let monomial = coef;
    for (const i of indices) monomial *= x[i];
    phi[r] = envelope * monomial;
    // d(phi_r)/dx_a = envelope * (d(monomial)/dx_a - x_a/sigma^2 * monomial)
    for (let a = 0; a < d; a++) {
      let dm = 0;
      for (let p = 0; p < indices.length; p++) {
        if (indices[p] !== a) continue;
        let v = coef;
        for (let q2 = 0; q2 < indices.length; q2++) if (q2 !== p) v *= x[indices[q2]];
        dm += v;
      }
      jac[r][a] = envelope * (dm - x[a] * inv * inv * monomial);
    }
  }
  return {
    phi,
    jac,
    derivative: precision === 2 ? 'exact-taylor-v2' : 'exact-taylor-v1',
  };
}

/**
 * 草图点积路径的合同导数：d(phi(x).phi(z))/dx = Jac(x)^T phi(z)。
 * 返回梯度与导数来源标记；调用方必须在误差合同声明的路径内使用。
 */
export function sketchContractGradient(x, z, sigma, precision) {
  const {phi, jac, derivative} = featureMapWithJacobian(x, sigma, precision);
  if (phi.length !== z.length) {
    throw new RangeError('合同梯度要求同档、同维度特征');
  }
  const grad = new Float64Array(x.length);
  for (let r = 0; r < phi.length; r++) {
    const w = z[r];
    if (w === 0) continue;
    const row = jac[r];
    for (let a = 0; a < x.length; a++) grad[a] += w * row[a];
  }
  return {gradient: grad, derivative, tier: precision === 2 ? TIERS.P2 : TIERS.V1};
}

/** 按 code 取真实向量（仅供本地精确核对，数据不出设备）。 */
export function codeVector(code) {
  return kernelCode(code);
}
