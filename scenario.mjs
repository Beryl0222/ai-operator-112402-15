// 设备联调入口：本地跑完整阈值交接脚本并打印可审计痕迹。
// 用法：node scenario.mjs [--ledger path] [--v2]
//   --ledger path  判决账本落盘路径（缺省仅内存）
//   --v2           新麦侧仅交付 scaledV2 视图，演示不可比与漂移守卫
import {runScenario} from './lib/scenario.mjs';

const argv = process.argv.slice(2);
const ledgerIdx = argv.indexOf('--ledger');
const ledgerPath = ledgerIdx >= 0 ? argv[ledgerIdx + 1] : null;
const forceV2SideB = argv.includes('--v2');

const {trace, ledger, budget} = runScenario({ledgerPath, forceV2SideB});

console.log('=== 动作痕迹 ===');
for (const step of trace) console.log(JSON.stringify(step));
console.log('\n=== 账本（设备可查的历史判决）===');
for (const record of ledger) console.log(JSON.stringify(record));
console.log('\n=== 预算结账 ===');
console.log(JSON.stringify({
  budgetBytes: {used: budget.bytes, limit: budget.byteLimit},
  exactPairs: {decision: budget.exactPairs, query: budget.queryPairs, pairLimit: budget.pairLimit},
}));
