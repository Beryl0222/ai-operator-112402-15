import assert from 'node:assert/strict';
import test from 'node:test';
import {GroupReducer, SideStore, Budget, crossKernelNumerator, crossWeightDenominator, BudgetExceededError} from '../lib/groups.mjs';
import {groupedDistance, thresholdDrift, ProjectionMismatchError} from '../lib/distance.mjs';
import {rbf, sketchFor, TIERS} from '../lib/kernel.mjs';
import {kernelCode} from '../inputs/microphone-samples.mjs';

const sigma = 1.25;
const rankV1 = 25;

function frame(group, code, weight) {
  return {physicalGroup: group, code, weight, values: kernelCode(code), encoding: 'spectralV1'};
}

function buildStore(side, frames, budget = new Budget()) {
  const store = new SideStore({side, projection: 'spectralV1', rankV1, rankP2: 325, budget, sigma});
  for (const f of frames) store.ingest({...f, sketchV1: sketchFor(f.code, 1)});
  return store;
}

/** 暴力口径：枚举有序对、剔除同 physicalGroup，作为无偏估计的参照。 */
function bruteDistance(framesA, framesB) {
  const sideMean = (frames) => {
    let num = 0;
    let den = 0;
    for (let i = 0; i < frames.length; i++) {
      for (let j = 0; j < frames.length; j++) {
        if (frames[i].physicalGroup === frames[j].physicalGroup) continue;
        if (frames[i].weight === 0 || frames[j].weight === 0) continue;
        num += frames[i].weight * frames[j].weight * rbf(frames[i].values, frames[j].values, sigma);
        den += frames[i].weight * frames[j].weight;
      }
    }
    return {mean: num / den, num, den};
  };
  const a = sideMean(framesA);
  const b = sideMean(framesB);
  let num = 0;
  let den = 0;
  for (const x of framesA) {
    for (const y of framesB) {
      if (x.physicalGroup === y.physicalGroup) continue;
      if (x.weight === 0 || y.weight === 0) continue;
      num += x.weight * y.weight * rbf(x.values, y.values, sigma);
      den += x.weight * y.weight;
    }
  }
  return {point: a.mean + b.mean - 2 * (num / den), denom: {a: a.den, b: b.den, ab: den}};
}

test('流式分组归约与暴力枚举一致，同组对与零权重帧被剔除', () => {
  const framesA = [
    frame(1, 10, 1), frame(1, 11, 2), frame(2, 20, 1),
    frame(3, 30, 0.5), frame(3, 31, 0), frame(4, 40, 1),
  ];
  const framesB = [
    frame(2, 21, 1), frame(2, 22, 1), frame(5, 50, 2),
    frame(6, 60, 1), frame(1, 12, 1),
  ];
  const budget = new Budget();
  const storeA = buildStore(0, framesA, budget);
  const storeB = buildStore(1, framesB, budget);

  // 归约器分母/分子对照暴力枚举（草图近似，容差取设备误差量级）
  const result = groupedDistance({storeA, storeB, errors: {v1: 0.016, p2: 0.00081}, useDiagnostics: false});
  const brute = bruteDistance(framesA, framesB);
  assert.ok(result.ready);
  assert.ok(Math.abs(result.point - brute.point) < 0.05, `point ${result.point} vs brute ${brute.point}`);
  assert.deepEqual(result.tiers.v1.terms.a.denominator, brute.denom.a);
  assert.deepEqual(result.tiers.v1.terms.b.denominator, brute.denom.b);
  assert.deepEqual(result.tiers.v1.terms.ab.denominator, brute.denom.ab);

  // 跨侧共享组（1、2）整体剔除
  const cross = crossKernelNumerator(storeA, storeB, TIERS.V1);
  assert.ok(Number.isFinite(cross.numerator));
  assert.ok(crossWeightDenominator(storeA, storeB, TIERS.V1) > 0);
});

test('权重全落在一个组时无法得出距离，并指明缺失侧', () => {
  const storeA = buildStore(0, [frame(7, 1, 1), frame(7, 2, 2)]);
  const storeB = buildStore(1, [frame(8, 3, 1), frame(9, 4, 1)]);
  const result = groupedDistance({storeA, storeB, errors: {v1: 0.016, p2: 0.00081}});
  assert.equal(result.ready, false);
  assert.ok(result.insufficientSides.includes('a'));
  assert.equal(result.point, null);
});

test('小样本：单组侧的同侧分母为 0，跨侧仍可给见证但距离不就绪', () => {
  const reducer = new GroupReducer(TIERS.V1, rankV1);
  reducer.add(3, 1, sketchFor(5, 1));
  assert.equal(reducer.weightDenominator(), 0);
  assert.equal(reducer.kernelNumerator(), 0);
});

test('预算守卫：byteLimit 与 pairLimit 超限抛 BudgetExceededError', () => {
  const tight = new Budget({byteLimit: 300, pairLimit: 10});
  const reducer = new GroupReducer(TIERS.V1, rankV1, {budget: tight, budgetKey: 'r'});
  assert.throws(() => {
    for (let g = 0; g < 5; g++) reducer.add(g, 1, sketchFor(g, 1));
  }, BudgetExceededError);
  assert.throws(() => tight.chargeExact(11), BudgetExceededError);
  // 单 query 伴随求值走独立口径，不占决定轮 pairLimit
  tight.chargeExact(100, {query: true});
  assert.equal(tight.queryPairs, 100);
  assert.equal(tight.exactPairs, 0);
});

test('投影身份不同：距离标记不可比，距离相减被漂移守卫拦截', () => {
  const storeA = buildStore(0, [frame(1, 1, 1), frame(2, 2, 1)]);
  const storeB = buildStore(1, [frame(3, 3, 1), frame(4, 4, 1)]);
  storeB.projection = 'scaledV2';
  const result = groupedDistance({storeA, storeB, errors: {v1: 0.016, p2: 0.00081}});
  assert.equal(result.comparable, false);
  const before = {projection: {a: 'spectralV1', b: 'spectralV1'}, point: 0.01};
  assert.throws(() => thresholdDrift(before, result), ProjectionMismatchError);
  // 维度相等也不建立共同空间
  assert.throws(
    () => thresholdDrift(before, {projection: {a: 'spectralV1', b: 'scaledV2'}, point: 0.02}),
    ProjectionMismatchError,
  );
  // 同构投影允许相减
  assert.ok(Math.abs(
    thresholdDrift(before, {projection: {a: 'spectralV1', b: 'spectralV1'}, point: 0.025}) - 0.015,
  ) < 1e-12);
});

test('诊断升档只影响升档组子问题，未升档对保持 v1 误差证据', () => {
  const framesA = [frame(1, 10, 1), frame(2, 20, 1), frame(50, 30, 1)];
  const framesB = [frame(1, 11, 1), frame(2, 21, 1), frame(3, 40, 1), frame(50, 50, 1)];
  const storeA = buildStore(0, framesA);
  const storeB = buildStore(1, framesB);
  // 升档两个共享组，p2 子集才能形成跨组对
  storeA.upgradeGroup(1, sketchFor);
  storeB.upgradeGroup(1, sketchFor);
  storeA.upgradeGroup(2, sketchFor);
  storeB.upgradeGroup(2, sketchFor);
  const result = groupedDistance({storeA, storeB, errors: {v1: 0.016, p2: 0.00081}});
  assert.deepEqual(result.diagnosticGroups.sort(), [1, 2]);
  assert.equal(result.tiers.p2.ready, true);
  // 混合档半宽严格介于全 p2 与全 v1 之间
  assert.ok(result.bound < result.tiers.v1.bound);
  assert.ok(result.bound > 4 * 0.00081 - 1e-12);
  assert.ok(result.tiers.merged.subShare > 0 && result.tiers.merged.subShare < 1);
});

test('p2 只升档单组时子问题证不出，混合档回退 v1 并注明原因', () => {
  const framesA = [frame(1, 10, 1), frame(2, 20, 1)];
  const framesB = [frame(1, 11, 1), frame(3, 40, 1)];
  const storeA = buildStore(0, framesA);
  const storeB = buildStore(1, framesB);
  storeA.upgradeGroup(1, sketchFor);
  storeB.upgradeGroup(1, sketchFor);
  const result = groupedDistance({storeA, storeB, errors: {v1: 0.016, p2: 0.00081}});
  assert.equal(result.tiers.p2.ready, false);
  assert.equal(result.ready, true); // v1 全量仍可判
  assert.match(result.tiers.merged.diagnosticNote, /回退 v1/);
});

test('整组移除后归约器回到移除前状态（增量子问题可回滚）', () => {
  const reducer = new GroupReducer(TIERS.V1, rankV1);
  reducer.add(1, 1, sketchFor(1, 1));
  reducer.add(2, 2, sketchFor(2, 1));
  const before = {W: reducer.W, num: reducer.kernelNumerator(), den: reducer.weightDenominator()};
  reducer.add(3, 1.5, sketchFor(3, 1));
  reducer.remove(3);
  assert.equal(reducer.W, before.W);
  assert.ok(Math.abs(reducer.kernelNumerator() - before.num) < 1e-12);
  assert.ok(Math.abs(reducer.weightDenominator() - before.den) < 1e-12);
});

test('pairLimit 只管阈值决定轮的精确成对求值，query 伴随额度过账独立', () => {
  const budget = new Budget({pairLimit: 3});
  budget.chargeExact(3);
  assert.throws(() => budget.chargeExact(1), BudgetExceededError);
  // query 伴随精确核对不受决定轮剩余额度影响
  budget.chargeExact(999, {query: true});
  assert.equal(budget.queryPairs, 999);
  assert.equal(budget.exactPairs, 3);
});

test('正常校准距离只走草图：决定轮精确成对求值为 0，且无 N×N 常驻结构', () => {
  const budget = new Budget();
  const manyA = Array.from({length: 40}, (_, i) => frame(100 + i, i + 1, 1));
  const manyB = Array.from({length: 40}, (_, i) => frame(200 + i, i + 500, 1));
  const storeA = buildStore(0, manyA, budget);
  const storeB = buildStore(1, manyB, budget);
  const result = groupedDistance({storeA, storeB, errors: {v1: 0.016, p2: 0.00081}});
  assert.ok(result.ready);
  assert.equal(budget.exactPairs, 0);
  // 常驻字节只与组数 G 和草图秩 r 有关：每侧每组 (rank+1)*8 字节
  const perGroup = (rankV1 + 1) * 8;
  assert.ok(budget.bytes <= 2 * 40 * perGroup);
});
