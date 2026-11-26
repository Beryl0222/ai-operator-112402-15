import assert from 'node:assert/strict';
import test from 'node:test';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {mkdtempSync, existsSync} from 'node:fs';
import {runScenario, loadSpec} from '../lib/scenario.mjs';
import {VERDICT} from '../lib/handoff.mjs';

test('联调脚本覆盖 actions 全序列且每步留下可审计痕迹', () => {
  const spec = loadSpec();
  const {trace, budget, ledger} = runScenario();
  assert.equal(trace.length, spec.actions.length);
  assert.deepEqual(trace.map((t) => t.action), spec.actions.map(([a]) => a));
  // 无 N×N 常驻：暂存字节始终在 budgetBytes 内
  assert.ok(budget.bytes <= spec.budgetBytes);
  // 决定轮不做全样本精确配对
  assert.equal(budget.exactPairs, 0);
  // 单 query 伴随精确求值单独计
  assert.ok(budget.queryPairs > 0);
  assert.ok(budget.queryPairs < spec.pairLimit);
  // 账本含旧麦历史判决
  assert.ok(ledger.some((r) => r.deviceId === 'M1' && r.verdict === VERDICT.ACCEPT));
});

test('scaledV2 视图判不可比，距离相减被漂移守卫拦截，阈值保持', () => {
  const {trace} = runScenario();
  const judge = trace.find((t) => t.action === 'judge');
  assert.equal(judge.comparable, false);
  assert.equal(judge.verdict, VERDICT.HOLD);
  assert.equal(judge.driftGuard.blocked, true);
  assert.match(judge.driftGuard.reason, /投影身份不可比/);
});

test('临界区先升诊断档再判决；证不出则 hold 并公布缺口侧/组', () => {
  const {trace} = runScenario();
  const hold = trace.find((t) => t.action === 'hold');
  assert.ok(hold.upgradedGroups.length > 0);
  assert.ok(hold.unavailableCount > 0);
  assert.equal(hold.heldThreshold, true);
  assert.equal(hold.post, VERDICT.HOLD);
  // 预算顶满时未升档组也被明确公布
  assert.ok(Array.isArray(hold.budgetBlocked));
});

test('新麦补回录音组后只重算该组子问题，最终判决与预算留痕完整', () => {
  const {trace} = runScenario();
  const add = trace.find((t) => t.action === 'group-add');
  assert.deepEqual(add.added, [8]);
  assert.ok(add.upgradedNow.includes(8));
  const choose = trace.find((t) => t.action === 'choose');
  assert.ok(choose.interval.length === 2);
  assert.ok(['accept', 'reject', 'hold'].includes(choose.verdict));
  assert.ok(choose.budget.bytes <= loadSpec().budgetBytes);
});

test('账本可落盘并被旧设备会话重新读取', () => {
  const dir = mkdtempSync(join(tmpdir(), 'scenario-ledger-'));
  const path = join(dir, 'decisions.json');
  runScenario({ledgerPath: path});
  assert.ok(existsSync(path));
  const {ledger} = runScenario({ledgerPath: path});
  // 两次运行追加，旧麦 M1 的基线仍可查
  assert.ok(ledger.filter((r) => r.deviceId === 'M1').length >= 2);
});

test('见证步骤同时给出敏感度与两条导数路径标记', () => {
  const {trace} = runScenario();
  const w = trace.find((t) => t.action === 'witness');
  assert.ok(w.sensitivityCount > 0);
  assert.equal(w.exactDerivative, 'exact-rbf-analytic');
  assert.equal(w.contractDerivative, 'exact-taylor-v1');
});
