// 设备联调场景驱动：按 inputs 里的 actions 顺序演示完整阈值交接流程。
// 两侧样本都在本地读取并归约，音频与特征不出设备。

import {readFileSync} from 'node:fs';
import {kernelRow} from '../inputs/microphone-samples.mjs';
import {
  Budget,
  SideStore,
  CalibrationSession,
  DecisionLedger,
  VERDICT,
  ProjectionMismatchError,
  thresholdDrift,
  exactWitnessGradient,
  sketchWitnessGradient,
  sketchFor,
} from './index.mjs';

export function loadSpec() {
  return JSON.parse(readFileSync(new URL('../inputs/microphone-samples.json', import.meta.url), 'utf8'));
}

/** 从 kernelRow 批量构建单侧样本库。projection 决定该侧是否具备共同投影身份。 */
export function buildSide(side, projection, spec, budget, {start = 0, count = null, rows = null} = {}) {
  const n = count ?? spec.rows[side];
  const store = new SideStore({
    side,
    projection,
    rankV1: spec.approximate.rank,
    rankP2: spec.precision2.rank,
    budget,
    sigma: spec.sigma,
  });
  const indices = rows ?? Array.from({length: Math.min(n, spec.rows[side] - start)}, (_, i) => start + i);
  for (const index of indices) {
    const row = kernelRow(side, index, projection);
    const sketchV1 = projection === spec.projection[0] ? sketchFor(row.code, 1) : null;
    store.ingest({...row, sketchV1});
  }
  return store;
}

/**
 * 按 actions 跑完整联调脚本，返回每一步的可审计痕迹。
 * @param {object} [opts]
 * @param {string} [opts.ledgerPath] 账本落盘路径；缺省只在内存
 * @param {boolean} [opts.forceV2SideB] 强制 B 侧仅交付 scaledV2 视图（默认按 actions 的 judge 步骤演示）
 */
export function runScenario({ledgerPath = null, forceV2SideB = false} = {}) {
  const spec = loadSpec();
  const budget = new Budget({byteLimit: spec.budgetBytes, pairLimit: spec.pairLimit});
  const ledger = new DecisionLedger(ledgerPath);
  const trace = [];

  // 预放一条旧麦历史判决，验证“设备用旧麦时仍可查到之前的判决和数据支持”。
  ledger.append({
    deviceId: 'M1',
    action: 'judge',
    verdict: VERDICT.ACCEPT,
    reason: 'historical-baseline',
    point: 0.0091,
    interval: [0.0041, 0.0141],
    projection: {a: spec.projection[0], b: spec.projection[0]},
  });

  let session = null;
  let stores = null;

  for (const [action, arg] of spec.actions) {
    if (action === 'renew') {
      // 新麦标定会话：A=旧麦 spectralV1，B=新麦（默认 spectralV1；可切 scaledV2）
      const projectionB = forceV2SideB ? spec.projection[2] : spec.projection[1];
      const storeA = buildSide(0, spec.projection[0], spec, budget);
      const storeB = buildSide(1, projectionB, spec, budget);
      stores = {storeA, storeB};
      session = new CalibrationSession({
        deviceId: arg,
        storeA,
        storeB,
        spec,
        budget,
        ledger,
      });
      trace.push({action, device: arg, projections: {a: storeA.projection, b: storeB.projection}});
    } else if (action === 'open') {
      trace.push({
        action,
        domain: arg,
        groups: {a: session.storeA.v1.positiveGroups().length, b: session.storeB.v1.positiveGroups().length},
        budgetBytes: budget.bytes,
      });
    } else if (action === 'judge') {
      // arg 给出本步要核验的投影身份；scaledV2 缺共同坐标时直接判不可比
      if (arg === spec.projection[2]) {
        const v2 = runIncomparableCheck({spec, budget, ledger, baseline: ledger.latest('M1')});
        trace.push({action, projection: arg, ...v2});
        continue;
      }
      const {decision, result} = session.judge({note: `projection-check:${arg}`});
      const step = {action, projection: arg, verdict: decision.verdict, reason: decision.reason};
      if (result.comparable && result.ready) {
        step.point = result.point;
        step.interval = result.interval;
        step.bound = result.bound;
      } else if (!result.comparable) {
        step.driftBlocked = true;
        step.driftGuard = guardDrift(ledger.latest('M1'), result);
      } else {
        step.insufficientSides = result.insufficientSides;
      }
      trace.push(step);
    } else if (action === 'hold') {
      // 临界区：先尝试诊断升档，再判一次；仍证不出就保留原阈值并公布缺口
      const before = session.judge({note: 'pre-diagnostic'});
      const upgrade = session.upgradeDiagnostics();
      const after = session.judge({note: 'post-diagnostic'});
      trace.push({
        action,
        group: arg,
        pre: before.decision.verdict,
        upgradedGroups: upgrade.upgraded.map((u) => u.group),
        budgetBlocked: upgrade.budgetBlocked.map((u) => u.group),
        unavailableCount: upgrade.unavailable.length,
        unavailableSample: upgrade.unavailable.slice(0, 5),
        post: after.decision.verdict,
        postReason: after.decision.reason,
        heldThreshold: after.decision.verdict === VERDICT.HOLD,
        interval: after.decision.interval,
      });
    } else if (action === 'witness') {
      // 单 query 核见证 + 对单组去留的敏感度 + 两条梯度路径
      const group = arg;
      const row = kernelRow(0, findIndexByGroup(0, group));
      const w1 = session.witness({queryCode: row.code, queryGroup: group, side: 0, precision: 1});
      const framesA = session.storeA.exactFrames();
      const framesB = session.storeB.exactFrames();
      const exact = exactWitnessGradient({
        framesA,
        framesB,
        query: row.values,
        queryGroup: group,
        side: 0,
        sigma: spec.sigma,
        budget,
      });
      const contract = sketchWitnessGradient({
        storeA: session.storeA,
        storeB: session.storeB,
        queryValues: row.values,
        queryGroup: group,
        side: 0,
        sigma: spec.sigma,
        precision: 1,
      });
      trace.push({
        action,
        group,
        witness: w1.ready ? w1.witness : null,
        sensitivityCount: w1.ready ? w1.sensitivity.length : 0,
        exactDerivative: exact.derivative,
        contractDerivative: contract.derivative,
        exactPairs: budget.queryPairs,
      });
    } else if (action === 'group-add') {
      // 新麦短暂补回一个录音组：只重算该组影响的子问题。
      // 合成数据的组号从 10 起；组 8 表示设备短暂补回的新组，按新麦侧码本造一帧。
      const group = arg;
      const existing = findIndexByGroup(1, group);
      const rows = existing === null
        ? [synthesizeGroupRow(1, group, spec)]
        : [kernelRow(1, existing, spec.projection[1])];
      const add = session.addGroup(1, rows);
      const rejudge = session.judge({note: `after-group-add:${group}`});
      trace.push({
        action,
        group,
        added: add.added,
        upgradedNow: add.upgradedNow,
        verdict: rejudge.decision.verdict,
        interval: rejudge.decision.interval,
      });
    } else if (action === 'choose') {
      const finalDecision = session.judge({note: 'final-hand-off'});
      trace.push({
        action,
        device: arg,
        verdict: finalDecision.decision.verdict,
        reason: finalDecision.decision.reason,
        interval: finalDecision.decision.interval,
        budget: {bytes: budget.bytes, exactPairs: budget.exactPairs, queryPairs: budget.queryPairs},
      });
    }
  }

  return {trace, ledger: ledger.records, budget};
}

function guardDrift(historical, result) {
  try {
    thresholdDrift(
      {projection: historical.projection, point: historical.point},
      {projection: result.projection, point: result.point},
    );
    return {blocked: false};
  } catch (err) {
    if (err instanceof ProjectionMismatchError) {
      return {blocked: true, reason: err.message};
    }
    throw err;
  }
}

function findIndexByGroup(side, group) {
  const spec = loadSpec();
  for (let i = 0; i < spec.rows[side]; i++) {
    if (kernelRow(side, i).physicalGroup === group) return i;
  }
  return null;
}

/** 设备短暂补回的录音组：沿用码本取真实向量与草图，物理组标记为补回组。 */
function synthesizeGroupRow(side, group, spec) {
  const code = (group * 7919 + side * 104729) % spec.approximate.codebook;
  const base = kernelRow(side, code % spec.rows[side], spec.projection[1]);
  return {...base, physicalGroup: group, code, encoding: spec.projection[1]};
}

/**
 * 第二代录音仅交付 scaledV2 视图：维度相等不建立共同空间。
 * 该侧没有可核验草图，距离判不可比、阈值保持，且距离相减被漂移守卫拦截。
 */
function runIncomparableCheck({spec, budget, ledger, baseline}) {
  const storeA = buildSide(0, spec.projection[0], spec, budget, {count: 128});
  const storeB = buildSide(1, spec.projection[2], spec, budget, {count: 128});
  const v2Session = new CalibrationSession({deviceId: 'M2-v2', storeA, storeB, spec, budget, ledger});
  const {decision, result} = v2Session.judge({note: 'scaledV2-only-view'});
  return {
    verdict: decision.verdict,
    reason: decision.reason,
    comparable: result.comparable,
    projections: result.projection,
    v1GroupsWithoutSketch: {a: storeA.v1.positiveGroups().length, b: storeB.v1.positiveGroups().length},
    driftGuard: guardDrift(baseline, result),
  };
}
