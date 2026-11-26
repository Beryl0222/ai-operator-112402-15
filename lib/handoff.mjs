// 阈值交接编排：证据门限判决、诊断档升档需求、本地判决账本，以及录音组
// 增量补回时只重算该组真正影响的子问题。全程本地完成，不上传音频。

import {readFileSync, writeFileSync, existsSync, mkdirSync} from 'node:fs';
import {dirname} from 'node:path';
import {groupedDistance, ProjectionMismatchError} from './distance.mjs';
import {kernelWitness} from './witness.mjs';
import {Budget, SideStore} from './groups.mjs';
import {sketchFor, TIERS} from './kernel.mjs';

export {Budget, SideStore};

export const VERDICT = Object.freeze({
  ACCEPT: 'accept', // 距离上界 ≤ accept：保留当前配置/沿用触发阈值
  REJECT: 'reject', // 距离下界 ≥ reject：当前配置作废，需要重新标定
  HOLD: 'hold', // 临界区，证据不足：保留原阈值，等待核证据
});

/**
 * 门限判决。只有当前阈值离误差区足够远才能保留或作废当前配置：
 * 上界 ≤ accept 接受；下界 ≥ reject 作废；其余进入临界区。
 */
export function decide(distanceResult, {accept, reject}) {
  if (!distanceResult.comparable) {
    return {
      verdict: VERDICT.HOLD,
      reason: 'projection-incomparable',
      distance: null,
      interval: null,
    };
  }
  if (!distanceResult.ready) {
    return {
      verdict: VERDICT.HOLD,
      reason: 'insufficient-groups',
      distance: null,
      interval: null,
      insufficientSides: distanceResult.insufficientSides,
    };
  }
  const [lo, hi] = distanceResult.interval;
  if (hi <= accept) return {verdict: VERDICT.ACCEPT, reason: 'upper-bound-within-accept', distance: distanceResult.point, interval: [lo, hi]};
  if (lo >= reject) return {verdict: VERDICT.REJECT, reason: 'lower-bound-beyond-reject', distance: distanceResult.point, interval: [lo, hi]};
  return {verdict: VERDICT.HOLD, reason: 'critical-zone', distance: distanceResult.point, interval: [lo, hi]};
}

/**
 * 诊断需求：临界区需要读诊断档公布的较高精度特征，但设备只为相关录音组
 * 提供诊断档。返回每一侧还缺哪些组，以及哪些组根本不在设备诊断供应范围。
 */
export function diagnosticNeeds(storeA, storeB, {maxDiagnosticGroups, supplyGroupIds}) {
  const groupsA = new Set(storeA.v1.positiveGroups());
  const groupsB = new Set(storeB.v1.positiveGroups());
  const haveA = new Set(storeA.upgradedGroups());
  const haveB = new Set(storeB.upgradedGroups());
  const supply = new Set(supplyGroupIds);
  const relevant = new Set([...groupsA, ...groupsB]);
  const missing = [];
  for (const g of [...relevant].sort((a, b) => a - b)) {
    const needA = groupsA.has(g) && !haveA.has(g);
    const needB = groupsB.has(g) && !haveB.has(g);
    if (!needA && !needB) continue;
    missing.push({
      group: g,
      sides: [...(needA ? ['a'] : []), ...(needB ? ['b'] : [])],
      suppliable: supply.has(g),
    });
  }
  const suppliable = missing.filter((m) => m.suppliable).slice(0, maxDiagnosticGroups);
  const unavailable = missing.filter((m) => !m.suppliable || !suppliable.includes(m));
  return {missing, suppliable, unavailable, totalRelevant: relevant.size};
}

/** 追加式本地判决账本：旧麦仍可查到之前的判决和数据支持。 */
export class DecisionLedger {
  constructor(path) {
    this.path = path;
    this.records = path && existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : [];
  }

  append(record) {
    const entry = {at: new Date().toISOString(), ...record};
    this.records.push(entry);
    if (this.path) {
      mkdirSync(dirname(this.path), {recursive: true});
      writeFileSync(this.path, JSON.stringify(this.records, null, 2));
    }
    return entry;
  }

  /** 按设备（麦克风）查历史判决；旧麦换会话后仍可读到数据支持。 */
  forDevice(deviceId) {
    return this.records.filter((r) => r.deviceId === deviceId);
  }

  latest(deviceId) {
    const all = this.forDevice(deviceId);
    return all.length ? all[all.length - 1] : null;
  }
}

/**
 * 一次校准会话：持有两侧样本库、预算与误差合同，串联 录入→距离→判决→
 * 诊断升档→再判决→账本 的完整流程。
 */
export class CalibrationSession {
  constructor({
    deviceId,
    storeA,
    storeB,
    spec, // inputs/microphone-samples.json 的规格段
    budget = null,
    ledger = null,
    sketchLoader = sketchFor,
  }) {
    this.deviceId = deviceId;
    this.storeA = storeA;
    this.storeB = storeB;
    this.spec = spec;
    this.budget = budget ?? new Budget({byteLimit: spec.budgetBytes, pairLimit: spec.pairLimit});
    this.ledger = ledger;
    this.sketchLoader = sketchLoader;
    this.errors = {v1: spec.approximate.absoluteError, p2: spec.precision2.absoluteError};
  }

  /** v1 流：录入一帧（带共同投影身份才产生草图核项）。 */
  ingest(side, row) {
    const store = side === 0 ? this.storeA : this.storeB;
    const comparableEncoding = row.encoding === this.spec.projection[0];
    const sketchV1 = comparableEncoding && row.code !== undefined ? this.sketchLoader(row.code, 1) : null;
    store.ingest({...row, sketchV1});
  }

  distance(useDiagnostics = true) {
    return groupedDistance({
      storeA: this.storeA,
      storeB: this.storeB,
      errors: this.errors,
      useDiagnostics,
    });
  }

  judge({note = null} = {}) {
    const result = this.distance(true);
    const decision = decide(result, {accept: this.spec.accept, reject: this.spec.reject});
    let diagnosticGap = null;
    if (decision.verdict === VERDICT.HOLD && result.comparable) {
      const needs = diagnosticNeeds(this.storeA, this.storeB, {
        maxDiagnosticGroups: this.spec.precision2.maxDiagnosticGroups,
        supplyGroupIds: Array.from({length: this.spec.precision2.maxDiagnosticGroups}, (_, i) => i),
      });
      diagnosticGap = {
        // 诊断口明确公布还缺哪侧哪组
        missing: needs.missing.map((m) => ({group: m.group, sides: m.sides, suppliable: m.suppliable})),
        budgetBlocked: needs.suppliable
          .filter((m) => !this.storeA.upgradedGroups().includes(m.group) &&
            !this.storeB.upgradedGroups().includes(m.group))
          .map((m) => m.group),
      };
    }
    const record = this.ledger?.append({
      deviceId: this.deviceId,
      action: 'judge',
      verdict: decision.verdict,
      reason: decision.reason,
      point: decision.distance,
      interval: decision.interval,
      comparable: result.comparable,
      projections: result.projection,
      diagnosticGroups: result.diagnosticGroups,
      diagnosticGap,
      insufficientSides: result.ready ? undefined : result.insufficientSides,
      budget: {bytes: this.budget.bytes, exactPairs: this.budget.exactPairs, queryPairs: this.budget.queryPairs},
      note,
    });
    return {decision, result, record};
  }

  /**
   * 临界区升档：只对设备能供应、且在 maxDiagnosticGroups 内的相关组读 precision2。
   * 暂存预算顶满时停止升档，剩余组进入 budgetBlocked；仍证不出时上层保持原阈值。
   */
  upgradeDiagnostics() {
    const supplyGroupIds = Array.from({length: this.spec.precision2.maxDiagnosticGroups}, (_, i) => i);
    const needs = diagnosticNeeds(this.storeA, this.storeB, {
      maxDiagnosticGroups: this.spec.precision2.maxDiagnosticGroups,
      supplyGroupIds,
    });
    const upgraded = [];
    const budgetBlocked = [];
    for (const item of needs.suppliable) {
      try {
        if (item.sides.includes('a')) this.storeA.upgradeGroup(item.group, this.sketchLoader);
        if (item.sides.includes('b')) this.storeB.upgradeGroup(item.group, this.sketchLoader);
        upgraded.push(item);
      } catch (err) {
        if (err.name !== 'BudgetExceededError') throw err;
        budgetBlocked.push({group: item.group, sides: item.sides, reason: err.message});
      }
    }
    return {upgraded, budgetBlocked, unavailable: needs.unavailable, needs};
  }

  /**
   * 新麦短暂补回一个录音组：只把该组加入 v1 流（以及可供应时的 p2），
   * 归约器按组累加，历史组的核项不重算、不重新配对。
   */
  addGroup(side, rows) {
    const store = side === 0 ? this.storeA : this.storeB;
    const before = new Set(store.v1.positiveGroups());
    for (const row of rows) this.ingest(side, row);
    const added = store.v1.positiveGroups().filter((g) => !before.has(g));
    const supplyGroupIds = new Set(
      Array.from({length: this.spec.precision2.maxDiagnosticGroups}, (_, i) => i),
    );
    const upgradedNow = [];
    for (const g of added) {
      if (supplyGroupIds.has(g)) {
        store.upgradeGroup(g, this.sketchLoader);
        upgradedNow.push(g);
      }
    }
    return {added, upgradedNow};
  }

  /** 单 query 见证（含对单组去留的敏感度）；伴随求值单独记 query 预算。 */
  witness({queryCode, queryGroup, side, precision = 1}) {
    return kernelWitness({
      storeA: this.storeA,
      storeB: this.storeB,
      queryCode,
      queryGroup,
      side,
      precision,
      budget: this.budget,
      sketchLoader: this.sketchLoader,
    });
  }
}

export {ProjectionMismatchError, TIERS};
