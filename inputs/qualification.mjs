export function inputRow(input, row) {
  const [, width, channels] = input.sourceShape;
  return Float32Array.from({ length: width * channels }, (_, j) => {
    const value = Math.imul(j + row * width * channels + input.seed, 479001599) >>> 14;
    return (value % 33 - 16) / 16;
  });
}
