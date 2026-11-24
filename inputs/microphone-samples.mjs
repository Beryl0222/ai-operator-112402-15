import {readFileSync} from 'node:fs';
const data = JSON.parse(readFileSync(new URL('./microphone-samples.json', import.meta.url), 'utf8'));
export function kernelRow(side, index, encoding='spectralV1') {
  if (index<0||index>=data.rows[side]) throw new RangeError('录音组索引越界');
  const relation=data.overlap.find(p=>p[side]===index);
  const group=relation ? relation[2] : 10+((index*13+side*53)% (data.groups-10));
  const code=(index*53+side*97)%data.approximate.codebook;
  const values=kernelCode(code);
  if(encoding==='scaledV2') for(let i=0;i<values.length;i++) values[i]*=data.scaleV2[i%data.scaleV2.length];
  return {physicalGroup:group,code,weight:data.weight[index%data.weight.length],values,encoding};
}
export function kernelCode(code) {
  const angle=code*.00009587379924;
  const radius=[.25,.375,.5][code%3];
  return Float64Array.from({length:data.dimension},(_,i)=>Math.cos(angle*(i+1)+(i+1)*.38)*radius/Math.sqrt(data.dimension));
}
export function kernelSketch(code) {
  const point=kernelCode(code);
  const output=new Float64Array(1+point.length);
  const invSigma=1/data.sigma;
  const envelope=Math.exp(-point.reduce((a,x)=>a+x*x,0)*invSigma**2/2);
  output[0]=envelope;
  for(let i=0;i<point.length;i++) output[i+1]=envelope*point[i]*invSigma;
  return output;
}
