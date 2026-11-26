import assert from 'node:assert/strict';
import test from 'node:test';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {mkdtempSync} from 'node:fs';
import {
  decide,
  VERDICT,
  DecisionLedger,
  diagnosticNeeds,
  CalibrationSession,
  Budget,
  SideStore,
} from '../lib/handoff.mjs';
import {sketchFor} from '../lib/kernel.mjs';
import {kernelCode, kernelSketch} from '../inputs/microphone-samples.mjs';

const spec = {
  rows: [8, 8],
  dimension: 24,
  sigma: 1.25,
  budgetBytes: 196608,
  pairLimit: 49152,
  accept: 0.014,
  reject: 0.045,
  projection: ['spectralV1', 'spectralV1', 'scaledV2'],
  approximate: {rank: 25, codebook: 65536, key: 'small-v1', absoluteError: 0.016},
  precision2: {key: 'diagnostic-v1', rank: 325, absoluteError: 0.00081, queryBlockSize: 96, maxDiagnosticGroups: 40},
};

function mkStore(side, projection, groups) {
  const store = new SideStore({side, projection, rankV1: 25, rankP2: 325, budget: new Budget(), sigma: 1.25});
  groups.forEach((g, i) => {
    const code = (i + 1 + side * 100) % 65536;
    store.ingest({
      physicalGroup: g,
      code,
      weight: 1,
      values: kernelCode(code),
      encoding: projection,
      sketchV1: projection === 'spectralV1' ? kernelSketch(code, 1) : null,
    });
  });
  return store;
}

test('判决：上界 ≤ accept 接受，下界 ≥ reject 作废，临界区 hold', () => {
  const base = {comparable: true, ready: true, point: 0.01, interval: null, insufficientSides: []};
  assert.equal(
    decide({...base, interval: [0.005, 0.013]}, spec).verdict,
    VERDICT.ACCEPT,
  );
  assert.equal(
    decide({...base, point: 0.05, interval: [0.046, 0.054]}, spec).verdict,
    VERDICT.REJECT,
  );
  const hold = decide({...base, point: 0.02, interval: [0.01, 0.03]}, spec);
  assert.equal(hold.verdict, VERDICT.HOLD);
  assert.equal(hold.reason, 'critical-zone');
});

test('不可比投影与证据不足一律 hold 并说明原因', () => {
  assert.equal(decide({comparable: false, ready: true}, spec).reason, 'projection-incomparable');
  const insufficient = decide({comparable: true, ready: false, insufficientSides: ['a', 'ab']}, spec);
  assert.equal(insufficient.verdict, VERDICT.HOLD);
  assert.deepEqual(insufficient.insufficientSides, ['a', 'ab']);
});

test('诊断需求按侧列出缺失组，并区分设备可供应与不可供应', () => {
  const storeA = mkStore(0, 'spectralV1', [1, 2, 60]);
  const storeB = mkStore(1, 'spectralV1', [2, 3, 60]);
  const needs = diagnosticNeeds(storeA, storeB, {
    maxDiagnosticGroups: 40,
    supplyGroupIds: [0, 1, 2, 3, 4],
  });
  const byGroup = Object.fromEntries(needs.missing.map((m) => [m.group, m]));
  assert.deepEqual(byGroup[1].sides, ['a']);
  assert.deepEqual(byGroup[3].sides, ['b']);
  assert.deepEqual(byGroup[60].sides.sort(), ['a', 'b']);
  assert.equal(byGroup[60].suppliable, false);
  assert.ok(needs.unavailable.some((u) => u.group === 60));
});

test('账本追加落盘，旧设备可查历史判决与数据支持', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ledger-'));
  const path = join(dir, 'decisions.json');
  const ledger = new DecisionLedger(path);
  ledger.append({deviceId: 'M1', verdict: VERDICT.ACCEPT, point: 0.01, interval: [0.005, 0.013]});
  ledger.append({deviceId: 'M2', verdict: VERDICT.HOLD, point: 0.02});
  const reopened = new DecisionLedger(path);
  assert.equal(reopened.forDevice('M1').length, 1);
  assert.equal(reopened.latest('M2').verdict, VERDICT.HOLD);
  assert.equal(reopened.latest('M1').interval[1], 0.013);
});

test('会话端到端：同投影距离可判决，预算口径分开记录', () => {
  const budget = new Budget({byteLimit: spec.budgetBytes, pairLimit: spec.pairLimit});
  const storeA = mkStore(0, 'spectralV1', [1, 2, 3, 4]);
  const storeB = mkStore(1, 'spectralV1', [5, 6, 7, 8]);
  const session = new CalibrationSession({deviceId: 'M2', storeA, storeB, spec, budget, ledger: new DecisionLedger()});
  const {decision, result, record} = session.judge();
  assert.ok([VERDICT.ACCEPT, VERDICT.HOLD, VERDICT.REJECT].includes(decision.verdict));
  assert.ok(result.comparable);
  assert.equal(record.deviceId, 'M2');
  assert.equal(budget.exactPairs, 0); // 草图路径不产生精确成对求值
});

test('补回一个录音组只增量更新该组，历史组不重新配对', () => {
  const budget = new Budget({byteLimit: spec.budgetBytes, pairLimit: spec.pairLimit});
  const storeA = mkStore(0, 'spectralV1', [1, 2, 3]);
  const storeB = mkStore(1, 'spectralV1', [4, 5, 6]);
  const session = new CalibrationSession({deviceId: 'M2', storeA, storeB, spec, budget, ledger: new DecisionLedger()});
  const before = session.distance().point;
  const code = 4242;
  session.addGroup(1, [{
    physicalGroup: 7,
    code,
    weight: 1,
    values: kernelCode(code),
    encoding: 'spectralV1',
  }]);
  assert.deepEqual(storeB.v1.positiveGroups(), [4, 5, 6, 7]);
  const after = session.distance().point;
  assert.notEqual(before, after); // 该组真正影响的子问题被重算
  assert.equal(budget.exactPairs, 0);
});
