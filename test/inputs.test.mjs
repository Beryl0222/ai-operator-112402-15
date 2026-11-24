import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const d = JSON.parse(readFileSync(new URL('../inputs/session.json', import.meta.url), 'utf8'));

test("相位稀疏槽到共享参数有效", () => {
  assert.ok(d.phases.every(p=>p.slots.length===9&&p.slots.every(id=>id===null||id in d.shared)));
});

test("监督区梯度的高分辨率坐标和形状一致", () => {
  assert.ok(d.windows.every(w=>w.grad.length===w.region[3]-w.region[1]&&w.grad.every(r=>r.length===w.region[2]-w.region[0])));
});

test("多相位的共享槽与混精度容量竞争", () => {
  assert.ok(d.phases.filter(p=>p.slots.includes('a')).length===3&&d.precision.exactSlots===1);
});

test("混精度消费优先级覆盖各相位", () => {
  assert.ok(d.precision.priority.length===d.phases.length && d.precision.priority.every(a=>d.phases.some(p=>p.id.join()===a.join())));
});
