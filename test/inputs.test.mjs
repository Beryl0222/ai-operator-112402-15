import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import { kernelRow, kernelCode, kernelSketch } from '../inputs/microphone-samples.mjs';

const data=JSON.parse(readFileSync(new URL('../inputs/microphone-samples.json',import.meta.url),'utf8'));
test('核码本、录音特征和观测权重的格式与关联', () => {
  assert.equal(kernelRow(0,37).physicalGroup,kernelRow(1,120).physicalGroup);
  assert.equal(kernelRow(0,700).values.length,data.dimension);
  assert.ok(data.pairLimit<data.rows[0]*data.rows[1]/25);
  assert.ok(data.weight.includes(0));
  assert.ok(data.accept< data.reject);
  assert.notEqual(data.projection[1],data.projection[2]);
  let largestError=0;
  for(let a=0;a<data.approximate.codebook;a+=517){
    const x=kernelCode(a), u=kernelSketch(a);
    for(let b=0;b<data.approximate.codebook;b+=641){
      const y=kernelCode(b),v=kernelSketch(b);
      const dot=u.reduce((s,f,i)=>s+f*v[i],0);
      const distance=x.reduce((s,f,i)=>s+(f-y[i])**2,0);
      largestError=Math.max(largestError,Math.abs(dot-Math.exp(-distance/(2*data.sigma**2))));
    }
  }
  assert.ok(largestError>0 && largestError<=data.approximate.absoluteError);

  const t=0.5**2/data.sigma**2;
  assert.ok(Math.exp(t)*t*t/2<=data.approximate.absoluteError);
  assert.notDeepEqual(kernelRow(1,120).values,kernelRow(1,120,"scaledV2").values);
  assert.equal(kernelSketch(731,2).length,data.precision2.rank);
  assert.ok(Math.exp(t)*t*t*t/6<=data.precision2.absoluteError);
  assert.ok(data.precision2.maxDiagnosticGroups<data.groups);
});
