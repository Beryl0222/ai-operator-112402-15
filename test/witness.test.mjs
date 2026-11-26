import assert from 'node:assert/strict';
import test from 'node:test';
import {kernelWitness, exactWitnessGradient, sketchWitnessGradient} from '../lib/witness.mjs';
import {SideStore, Budget} from '../lib/groups.mjs';
import {rbf, TIERS} from '../lib/kernel.mjs';
import {kernelCode, kernelSketch} from '../inputs/microphone-samples.mjs';

const sigma = 1.25;
const rankV1 = 25;

function mkStore(side, frames) {
  const store = new SideStore({side, projection: 'spectralV1', rankV1, rankP2: 325, budget: new Budget(), sigma});
  for (const f of frames) {
    store.ingest({...f, sketchV1: kernelSketch(f.code, 1)});
  }
  return store;
}

const framesA = [
  {physicalGroup: 1, code: 10, weight: 1, values: kernelCode(10), encoding: 'spectralV1'},
  {physicalGroup: 1, code: 11, weight: 1, values: kernelCode(11), encoding: 'spectralV1'},
  {physicalGroup: 2, code: 20, weight: 2, values: kernelCode(20), encoding: 'spectralV1'},
  {physicalGroup: 3, code: 30, weight: 1, values: kernelCode(30), encoding: 'spectralV1'},
];
const framesB = [
  {physicalGroup: 4, code: 40, weight: 1, values: kernelCode(40), encoding: 'spectralV1'},
  {physicalGroup: 5, code: 50, weight: 1, values: kernelCode(50), encoding: 'spectralV1'},
  {physicalGroup: 2, code: 21, weight: 1, values: kernelCode(21), encoding: 'spectralV1'},
];

test('见证值 = 两侧剔除 query 组后的加权核均值差', () => {
  const storeA = mkStore(0, framesA);
  const storeB = mkStore(1, framesB);
  const queryCode = 10;
  const w = kernelWitness({
    storeA, storeB, queryCode, queryGroup: 1, side: 0, precision: 1, sketchLoader: kernelSketch,
  });
  assert.ok(w.ready);
  // 暴力参照：A 侧剔除组 1
  const wa = framesA.filter((f) => f.physicalGroup !== 1 && f.weight > 0);
  const meanA = wa.reduce((s, f) => s + f.weight * rbf(kernelCode(queryCode), f.values, sigma), 0)
    / wa.reduce((s, f) => s + f.weight, 0);
  const wb = framesB.filter((f) => f.weight > 0);
  const meanB = wb.reduce((s, f) => s + f.weight * rbf(kernelCode(queryCode), f.values, sigma), 0)
    / wb.reduce((s, f) => s + f.weight, 0);
  // 草图近似，容差取 v1 误差量级
  assert.ok(Math.abs(w.witness - (meanA - meanB)) < 0.02);
});

test('对单组去留敏感：逐组剔除结果可枚举，query 自身组在本侧已排除', () => {
  const storeA = mkStore(0, framesA);
  const storeB = mkStore(1, framesB);
  const w = kernelWitness({
    storeA, storeB, queryCode: 30, queryGroup: 3, side: 0, precision: 1, sketchLoader: kernelSketch,
  });
  const groupsCovered = new Set(w.sensitivity.map((s) => `${s.side}:${s.group}`));
  // 两侧所有正权重组都有一条敏感度记录
  for (const g of [1, 2, 3]) assert.ok(groupsCovered.has(`0:${g}`));
  for (const g of [2, 4, 5]) assert.ok(groupsCovered.has(`1:${g}`));
  assert.ok(w.sensitivity.every((s) => s.reason === undefined || s.witness !== undefined));
});

test('精确核见证梯度通过有限差分，且按 query 口径单独计精确对数', () => {
  const budget = new Budget({pairLimit: 2}); // 决定轮额度极小
  const query = kernelCode(30);
  const out = exactWitnessGradient({
    framesA, framesB, query, queryGroup: 3, side: 0, sigma, budget,
  });
  assert.ok(out.ready);
  assert.equal(out.derivative, 'exact-rbf-analytic');
  const eps = 1e-6;
  for (let d = 0; d < query.length; d += 4) {
    const qp = Float64Array.from(query);
    const qm = Float64Array.from(query);
    qp[d] += eps;
    qm[d] -= eps;
    const fp = exactWitnessGradient({framesA, framesB, query: qp, queryGroup: 3, side: 0, sigma}).witness;
    const fm = exactWitnessGradient({framesA, framesB, query: qm, queryGroup: 3, side: 0, sigma}).witness;
    const fd = (fp - fm) / (2 * eps);
    assert.ok(Math.abs(fd - out.gradient[d]) < 1e-6, `dim ${d}: ${fd} vs ${out.gradient[d]}`);
  }
  // query 伴随求值不占决定轮 pairLimit
  assert.equal(budget.exactPairs, 0);
  assert.ok(budget.queryPairs > 0);
});

test('对训练态帧输入的梯度同样通过有限差分', () => {
  const target = framesA.find((f) => f.physicalGroup === 2).values;
  const query = kernelCode(30);
  const out = exactWitnessGradient({
    framesA, framesB, query, queryGroup: 3, side: 0, sigma, trainable: target,
  });
  const eps = 1e-6;
  for (let d = 0; d < target.length; d += 6) {
    const saved = target[d];
    target[d] = saved + eps;
    const fp = exactWitnessGradient({framesA, framesB, query, queryGroup: 3, side: 0, sigma, trainable: target}).witness;
    target[d] = saved - eps;
    const fm = exactWitnessGradient({framesA, framesB, query, queryGroup: 3, side: 0, sigma, trainable: target}).witness;
    target[d] = saved;
    const fd = (fp - fm) / (2 * eps);
    assert.ok(Math.abs(fd - out.gradient[d]) < 1e-6, `dim ${d}`);
  }
});

test('草图合同梯度标注导数来源，且与合同映射雅可比一致', () => {
  const storeA = mkStore(0, framesA);
  const storeB = mkStore(1, framesB);
  const out = sketchWitnessGradient({
    storeA, storeB, queryValues: kernelCode(30), queryGroup: 3, side: 0, sigma, precision: 1,
  });
  assert.ok(out.ready);
  assert.equal(out.derivative, 'exact-taylor-v1');
  assert.equal(out.tier, TIERS.V1);
  assert.equal(out.gradient.length, 24);
  // 有限差分对照草图见证（映射对 query 的导数）
  const eps = 1e-6;
  const q = kernelCode(30);
  for (let d = 0; d < q.length; d += 8) {
    const qp = Float64Array.from(q);
    const qm = Float64Array.from(q);
    qp[d] += eps;
    qm[d] -= eps;
    const witnessAt = (vec) => sketchScalarWitness(storeA, storeB, vec, 3, 0);
    const fd = (witnessAt(qp) - witnessAt(qm)) / (2 * eps);
    assert.ok(Math.abs(fd - out.gradient[d]) < 1e-5, `dim ${d}: ${fd} vs ${out.gradient[d]}`);
  }
});

function sketchScalarWitness(storeA, storeB, queryValues, group, side) {
  // 用一阶 Taylor 映射在本地重建见证标量
  const inv = 1 / sigma;
  const envelope = Math.exp(-queryValues.reduce((a, x) => a + x * x, 0) * inv * inv / 2);
  const phi = new Float64Array(1 + queryValues.length);
  phi[0] = envelope;
  for (let i = 0; i < queryValues.length; i++) phi[i + 1] = envelope * queryValues[i] * inv;
  const mean = (store, excluded) => {
    const {W, S} = store.v1.aggregateWithout(excluded);
    let s = 0;
    for (let r = 0; r < phi.length; r++) s += S[r] * phi[r];
    return s / W;
  };
  return mean(storeA, side === 0 ? group : -1) - mean(storeB, side === 1 ? group : -1);
}
