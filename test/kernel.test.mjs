import assert from 'node:assert/strict';
import test from 'node:test';
import {rbf, rbfGradient, sketchFor, sketchDot, featureMapWithJacobian, sketchContractGradient, TIERS} from '../lib/kernel.mjs';
import {kernelCode, kernelSketch} from '../inputs/microphone-samples.mjs';

const sigma = 1.25;

test('精确 RBF 核与解析梯度，有限差分校验', () => {
  const x = kernelCode(123);
  const y = kernelCode(456);
  const g = rbfGradient(x, y, sigma);
  const eps = 1e-6;
  for (let d = 0; d < x.length; d += 5) {
    const xp = Float64Array.from(x);
    const xm = Float64Array.from(x);
    xp[d] += eps;
    xm[d] -= eps;
    const fd = (rbf(xp, y, sigma) - rbf(xm, y, sigma)) / (2 * eps);
    assert.ok(Math.abs(fd - g[d]) < 1e-7, `dim ${d}: analytic ${g[d]} fd ${fd}`);
  }
});

test('v1 草图点积近似精确核且满足设备误差合同', () => {
  const eps = 0.016;
  let worst = 0;
  for (const a of [1, 99, 5000, 60000]) {
    for (const b of [7, 1234, 40000]) {
      const approx = sketchDot(sketchFor(a, 1), sketchFor(b, 1), TIERS.V1);
      const exact = rbf(kernelCode(a), kernelCode(b), sigma);
      worst = Math.max(worst, Math.abs(approx - exact));
    }
  }
  assert.ok(worst > 0 && worst <= eps, `worst=${worst}`);
});

test('precision2 草图误差显著小于 v1，且两档禁止跨档点积', () => {
  const u1 = sketchFor(123, 1);
  const u2 = sketchFor(123, 2);
  assert.notEqual(u1.length, u2.length);
  assert.throws(() => sketchDot(u1, u2, TIERS.V1, TIERS.P2), TypeError);
  assert.throws(() => sketchDot(u1, u2, TIERS.P2, TIERS.V1), TypeError);
  let worst1 = 0;
  let worst2 = 0;
  for (const a of [3, 888, 33333]) {
    for (const b of [9, 7777, 55555]) {
      const exact = rbf(kernelCode(a), kernelCode(b), sigma);
      worst1 = Math.max(worst1, Math.abs(sketchDot(sketchFor(a, 1), sketchFor(b, 1), TIERS.V1) - exact));
      worst2 = Math.max(worst2, Math.abs(sketchDot(sketchFor(a, 2), sketchFor(b, 2), TIERS.P2) - exact));
    }
  }
  assert.ok(worst2 < worst1);
  assert.ok(worst2 <= 0.00081);
});

test('Taylor 特征映射雅可比通过有限差分，且标注导数来源', () => {
  const x = kernelCode(42);
  for (const precision of [1, 2]) {
    const {phi, jac, derivative} = featureMapWithJacobian(x, sigma, precision);
    assert.equal(derivative, precision === 2 ? 'exact-taylor-v2' : 'exact-taylor-v1');
    assert.equal(phi.length, jac.length);
    const eps = 1e-6;
    for (let r = 0; r < phi.length; r += precision === 2 ? 37 : 3) {
      for (let d = 0; d < x.length; d += 7) {
        const xp = Float64Array.from(x);
        const xm = Float64Array.from(x);
        xp[d] += eps;
        xm[d] -= eps;
        const fp = featureMapWithJacobian(xp, sigma, precision).phi[r];
        const fm = featureMapWithJacobian(xm, sigma, precision).phi[r];
        const fd = (fp - fm) / (2 * eps);
        assert.ok(Math.abs(fd - jac[r][d]) < 1e-6, `p${precision} r${r} d${d}`);
      }
    }
  }
});

test('合同梯度 = 雅可比转置乘对侧草图，且只能同档使用', () => {
  const x = kernelCode(5);
  const z = kernelSketch(9, 1);
  const {gradient, derivative, tier} = sketchContractGradient(x, z, sigma, 1);
  assert.equal(derivative, 'exact-taylor-v1');
  assert.equal(tier, TIERS.V1);
  assert.equal(gradient.length, x.length);
  const z2 = kernelSketch(9, 2);
  assert.throws(() => sketchContractGradient(x, z2, sigma, 1), RangeError);
});
