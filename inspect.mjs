import {readFileSync} from 'node:fs';
import * as input from './inputs/microphone-samples.mjs';

const source=JSON.parse(readFileSync(new URL('./inputs/microphone-samples.json',import.meta.url),'utf8'));
console.log(JSON.stringify({名称:'核码本、录音特征和观测权重',首项:source['rows'],字段:Object.keys(source),取数函数:Object.keys(input)},null,2));
