// 分组流式归约：同 physicalGroup 的帧来自同一次播放（回声分离前后），
// 不能当作独立样本。归约器只常驻“按组的权重和与草图加权和”，
// 不常驻 N×N 核矩阵；核和用总量平方减去组内平方在线得到。

import {sketchFor, sketchDot, TIERS} from './kernel.mjs';

export class BudgetExceededError extends Error {
  constructor(kind, used, limit) {
    super(`预算超限：${kind} 已用 ${used}，上限 ${limit}`);
    this.name = 'BudgetExceededError';
    this.kind = kind;
    this.used = used;
    this.limit = limit;
  }
}

/**
 * 预算记账。
 * - bytes：当前常驻暂存数据的字节数（budgetBytes 口径）；
 * - exactPairs：典型阈值决定轮中真实精确成对核求值数（pairLimit 口径），
 *   单 query 的伴随精确求值由 queryPairs 单独计，不混入前一决定。
 */
export class Budget {
  constructor({byteLimit = Infinity, pairLimit = Infinity} = {}) {
    this.byteLimit = byteLimit;
    this.pairLimit = pairLimit;
    this.bytes = 0;
    this.exactPairs = 0;
    this.queryPairs = 0;
    this._allocs = new Map();
  }

  alloc(key, bytes) {
    if (this._allocs.has(key)) this.free(key);
    if (this.bytes + bytes > this.byteLimit) {
      throw new BudgetExceededError('budgetBytes', this.bytes + bytes, this.byteLimit);
    }
    this._allocs.set(key, bytes);
    this.bytes += bytes;
  }

  free(key) {
    const bytes = this._allocs.get(key);
    if (bytes === undefined) return;
    this.bytes -= bytes;
    this._allocs.delete(key);
  }

  chargeExact(n = 1, {query = false} = {}) {
    if (query) {
      this.queryPairs += n;
      return;
    }
    if (this.exactPairs + n > this.pairLimit) {
      throw new BudgetExceededError('pairLimit', this.exactPairs + n, this.pairLimit);
    }
    this.exactPairs += n;
  }
}

const F64 = 8;

/**
 * 单侧、单档的分组归约器。
 * 维护 G 个组的 w_g 与 s_g = sum_{i in g} w_i phi(x_i)，
 * 以及全局 W = sum w、S = sum w phi。
 * 同组核和（加权有序对）= W^2 - sum_g w_g^2；
 * 同组草图核和       = ||S||^2 - sum_g ||s_g||^2。
 */
export class GroupReducer {
  constructor(tier, rank, {budget = null, budgetKey = null} = {}) {
    this.tier = tier;
    this.rank = rank;
    this.groups = new Map(); // group -> {w, s:Float64Array, bytes}
    this.W = 0;
    this.S = new Float64Array(rank);
    this._budget = budget;
    this._budgetKey = budgetKey;
    this.bytes = 0;
  }

  _account(delta) {
    this.bytes += delta;
    if (this._budget) this._budget.alloc(this._budgetKey, this.bytes);
  }

  /** 加入一帧。weight 为 0 的帧不参与任何核项，也不产生梯度。 */
  add(group, weight, vector) {
    if (vector.length !== this.rank) {
      throw new RangeError(`草图维度 ${vector.length} 与归约器档位 ${this.tier}(${this.rank}) 不符`);
    }
    if (weight === 0) return;
    let slot = this.groups.get(group);
    if (!slot) {
      slot = {w: 0, s: new Float64Array(this.rank)};
      this.groups.set(group, slot);
      this._account(F64 + this.rank * F64);
    }
    slot.w += weight;
    this.W += weight;
    for (let r = 0; r < this.rank; r++) {
      const add = weight * vector[r];
      slot.s[r] += add;
      this.S[r] += add;
    }
  }

  /** 整组移除（新麦补回/撤下某录音组时只重算该组影响的子问题）。 */
  remove(group) {
    const slot = this.groups.get(group);
    if (!slot) return;
    this.W -= slot.w;
    for (let r = 0; r < this.rank; r++) this.S[r] -= slot.s[r];
    this.groups.delete(group);
    this._account(-(F64 + this.rank * F64));
  }

  positiveGroups() {
    return [...this.groups.keys()].filter((g) => this.groups.get(g).w > 0).sort((a, b) => a - b);
  }

  /** 跨组草图核和：||S||^2 - sum ||s_g||^2，可限定组子集。 */
  kernelNumerator(groupSet = null) {
    let total = 0;
    const useAll = groupSet === null;
    if (useAll) {
      for (const r of this.S) total += r * r;
    } else {
      for (const [g, slot] of this.groups) {
        if (!groupSet.has(g)) continue;
        for (const r of slot.s) total += r * r;
      }
    }
    let inner = 0;
    for (const [g, {s}] of this.groups) {
      if (!useAll && !groupSet.has(g)) continue;
      for (const r of s) inner += r * r;
    }
    return total - inner;
  }

  /** 有效权重对数，可限定组子集：(sum w_g)^2 - sum w_g^2。 */
  weightDenominator(groupSet = null) {
    let W = 0;
    let inner = 0;
    for (const [g, slot] of this.groups) {
      if (groupSet !== null && !groupSet.has(g)) continue;
      W += slot.w;
      inner += slot.w * slot.w;
    }
    return W * W - inner;
  }

  /** 剔除指定组后的权重与与加权草图和（见证的 leave-one-group-out）。 */
  aggregateWithout(excluded) {
    const slot = this.groups.get(excluded);
    const W = this.W - (slot ? slot.w : 0);
    const S = Float64Array.from(this.S);
    if (slot) for (let r = 0; r < this.rank; r++) S[r] -= slot.s[r];
    return {W, S};
  }
}

/**
 * 单侧样本库，持有两档归约器与投影身份。
 * v1 档始终随校准流维护；p2 档只在诊断轮为可从源端读回的组升档，
 * 设备只对相关录音组提供诊断档。
 */
export class SideStore {
  constructor({side, projection, rankV1, rankP2, budget, sigma}) {
    this.side = side;
    this.projection = projection;
    this.sigma = sigma;
    this.v1 = new GroupReducer(TIERS.V1, rankV1, {budget, budgetKey: `side${side}:v1`});
    this.p2 = new GroupReducer(TIERS.P2, rankP2, {budget, budgetKey: `side${side}:p2`});
    this._codes = new Map(); // group -> [{code, weight}]，诊断升档与精确核对用
    this._vectors = new Map(); // group -> [{values, weight}]，精确 query 梯度用
  }

  /**
   * 录入一帧。
   * - spectralV1 且带 code：可进入 v1 草图流；
   * - precision2 草图仅通过 upgradeGroup 显式升档；
   * - scaledV2 之类无可核验投影身份：只登记元数据，不产生草图核项。
   */
  ingest({physicalGroup, weight, code, values, encoding, sketchV1 = null}) {
    if (!this._codes.has(physicalGroup)) this._codes.set(physicalGroup, []);
    this._codes.get(physicalGroup).push({code, weight});
    if (!this._vectors.has(physicalGroup)) this._vectors.set(physicalGroup, []);
    this._vectors.get(physicalGroup).push({values, weight});
    if (sketchV1) this.v1.add(physicalGroup, weight, sketchV1);
  }

  /** 为某组升 precision2 档；需要该组全部帧的 code 以重建二阶草图。 */
  upgradeGroup(group, sketchLoader) {
    const frames = this._codes.get(group);
    if (!frames) throw new RangeError(`侧 ${this.side} 不存在录音组 ${group}，无法升档`);
    for (const {code, weight} of frames) {
      if (weight === 0) continue;
      this.p2.add(group, weight, sketchLoader(code, 2));
    }
  }

  downgradeGroup(group) {
    this.p2.remove(group);
  }

  upgradedGroups() {
    return this.p2.positiveGroups();
  }

  /** 该侧帧的精确向量（本地核对，不上传音频）。 */
  exactFrames() {
    const out = [];
    for (const [group, frames] of this._vectors) {
      for (const f of frames) out.push({group, ...f});
    }
    return out;
  }
}

/**
 * 跨侧草图核和，整体剔除共享物理组（同源组不能估计独立样本的组内期望）。
 * 对每对组 (g,h)，g != h 才计入：(sum_i w_i phi_i)·(sum_j w_j psi_j)。
 * 流式可维护；这里按组累加器配对，配对数是 G0*G1 而非 N0*N1。
 */
/**
 * 跨侧草图核和，整体剔除共享物理组（同源组不能估计独立样本的组内期望）。
 * 组聚合向量已是加权和，组间直接点积；g==h 的共享同源组剔除。
 * groupSet 非空时只统计两侧都在集合内的组（诊断升档子集）。
 */
export function crossKernelNumerator(storeA, storeB, tier, groupSet = null) {
  const ra = tier === TIERS.P2 ? storeA.p2 : storeA.v1;
  const rb = tier === TIERS.P2 ? storeB.p2 : storeB.v1;
  let numerator = 0;
  let sameGroupKernel = 0;
  for (const [g, sa] of ra.groups) {
    if (groupSet && !groupSet.has(g)) continue;
    for (const [h, tb] of rb.groups) {
      if (groupSet && !groupSet.has(h)) continue;
      const dot = sketchDot(sa.s, tb.s, ra.tier, rb.tier);
      if (g !== h) numerator += dot;
      else sameGroupKernel += dot;
    }
  }
  return {numerator, sameGroupKernel};
}

/** 跨侧权重分母：W_A W_B - sum_g w_A,g w_B,g（共享组整体剔除），可限定组子集。 */
export function crossWeightDenominator(storeA, storeB, tier, groupSet = null) {
  const ra = tier === TIERS.P2 ? storeA.p2 : storeA.v1;
  const rb = tier === TIERS.P2 ? storeB.p2 : storeB.v1;
  let wa = 0;
  let wb = 0;
  let shared = 0;
  for (const [g, sa] of ra.groups) {
    if (groupSet && !groupSet.has(g)) continue;
    wa += sa.w;
    const sb = rb.groups.get(g);
    if (sb && (!groupSet || groupSet.has(g))) {
      wb += sb.w;
      shared += sa.w * sb.w;
    }
  }
  if (groupSet) {
    for (const [h, sb] of rb.groups) {
      if (groupSet.has(h) && !ra.groups.has(h)) wb += sb.w;
    }
  } else {
    wb = rb.W;
  }
  return wa * wb - shared;
}
