export interface SamplingOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  seenIds?: readonly number[];
}

export function needsSamplingReadback(sampling: SamplingOptions | undefined): boolean {
  if (!sampling) return false;
  return (
    (sampling.temperature ?? 0) > 0 ||
    (sampling.topP ?? 1) < 1 ||
    normalizeTopK(sampling.topK) > 0
  );
}

export function needsCandidateReadback(sampling: SamplingOptions | undefined): boolean {
  if (!sampling) return false;
  return needsSamplingReadback(sampling) || (sampling.repetitionPenalty ?? 1) !== 1;
}

export function sampleFromCandidateBuffer(
  buffer: ArrayBuffer,
  sampling: SamplingOptions | undefined,
  candidateCount = 256,
): number {
  const bytes = buffer.slice(0);
  const floats = new Float32Array(bytes);
  const ids = new Uint32Array(bytes);
  const seen = new Set(sampling?.seenIds ?? []);
  const repetitionPenalty = Math.max(sampling?.repetitionPenalty ?? 1, 1e-6);
  const candidates: Array<{ val: number; id: number }> = [];
  const n = Math.min(candidateCount, Math.floor(floats.length / 2));

  for (let i = 0; i < n; i++) {
    let val = floats[i * 2];
    const id = ids[i * 2 + 1];
    if (!Number.isFinite(val) || id === 0xffffffff) continue;
    if (repetitionPenalty !== 1 && seen.has(id)) {
      val = val > 0 ? val / repetitionPenalty : val * repetitionPenalty;
    }
    candidates.push({ val, id });
  }

  candidates.sort((a, b) => b.val - a.val);
  const topK = normalizeTopK(sampling?.topK);
  const limited = topK > 0 ? candidates.slice(0, Math.min(topK, candidates.length)) : candidates;
  const temperature = sampling?.temperature ?? 0;
  if (temperature <= 0) return limited[0]?.id ?? candidates[0]?.id ?? 0;

  const topP = Math.min(Math.max(sampling?.topP ?? 1, 0), 1);
  const maxVal = limited[0]?.val ?? 0;
  const probs = limited.map((c) => Math.exp((c.val - maxVal) / temperature));
  const total = probs.reduce((a, b) => a + b, 0);
  let cutoff = limited.length;
  if (topP < 1 && total > 0) {
    let cdf = 0;
    for (let i = 0; i < probs.length; i++) {
      cdf += probs[i] / total;
      if (cdf >= topP) {
        cutoff = i + 1;
        break;
      }
    }
  }

  let sampleTotal = 0;
  for (let i = 0; i < cutoff; i++) sampleTotal += probs[i];
  let r = Math.random() * sampleTotal;
  for (let i = 0; i < cutoff; i++) {
    r -= probs[i];
    if (r <= 0) return limited[i].id;
  }
  return limited[Math.max(0, cutoff - 1)]?.id ?? limited[0]?.id ?? 0;
}

function normalizeTopK(topK: number | undefined): number {
  if (!Number.isFinite(topK) || topK == null) return 0;
  return Math.max(0, Math.floor(topK));
}
