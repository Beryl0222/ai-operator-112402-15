import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

const qualification = JSON.parse(readFileSync(new URL('../inputs/qualification.json', import.meta.url), 'utf8'));

const { inputRow } = await import('../inputs/qualification.mjs');
test('设备校准覆盖不同布局并引用同一物理权重', () => {
  for (const item of qualification.calibration) {
    assert.deepEqual(Object.keys(item.signedSamples).sort(),
      qualification.deviceMask[item.device][item.phase].slice().sort());
    assert.ok(Object.keys(item.signedSamples).every((id) => Object.hasOwn(qualification.weights, id)));
  }
  const [one, two] = Object.values(qualification.deviceMask);
  assert.ok(one[0].includes('a') && two[0].includes('a'));
  assert.notDeepEqual(one, two);
  assert.equal(inputRow(qualification, 10).length, 496 * 8);
  assert.ok(inputRow(qualification, 10).some((value) => value < 0));
});
test('重复监督、旧修订与受限输出张量来自同一个训练回合', () => {
  const [, w] = qualification.sourceShape;
  const windows = Object.fromEntries(qualification.windows.map((window) => [window.id, window]));
  const first = qualification.events.findIndex((event) => event[0] === 'loss');
  const before = qualification.events[first + 1];
  assert.deepEqual(before, qualification.events[first]);
  assert.equal(windows[before[1]].revision, windows.q.revision);
  assert.equal(qualification.events[first + 2][1], windows.q.device);
  assert.notEqual(qualification.events[first + 2][1], windows.r.device);
  assert.equal(windows.r.region[2], w * 2);
  for (const { region: [left, top, right, bottom] } of qualification.windows) {
    assert.ok((right - left) * (bottom - top) * 4 > qualification.maxWorkBytes);
  }
});
