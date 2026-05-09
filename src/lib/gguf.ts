import { fetchCachedRange, type ProgressCallback } from "./cache";

export const GGUF_MAGIC = 0x46554747; // "GGUF", little-endian

export const GGML_TYPE = {
  F32: 0,
  F16: 1,
  Q5_0: 6,
  Q5_1: 7,
  Q8_0: 8,
  Q4_K: 12,
  Q5_K: 13,
  Q6_K: 14,
} as const;

export type GgmlType = (typeof GGML_TYPE)[keyof typeof GGML_TYPE];

export interface GgufValue {
  type: string;
  value: unknown;
}

export interface GgufTensorInfo {
  name: string;
  shape: number[];
  type: number;
  offset: number;
  nElements: number;
  nBytes: number;
}

export interface GgufFile {
  version: number;
  tensorCount: number;
  kvCount: number;
  kv: Map<string, GgufValue>;
  tensors: GgufTensorInfo[];
  tensorMap: Map<string, GgufTensorInfo>;
  dataOffset: number;
}

export async function fetchGgufHeader(url: string, onProgress?: ProgressCallback): Promise<GgufFile> {
  let size = 4 * 1024 * 1024;
  let lastError: unknown = null;
  while (size <= 64 * 1024 * 1024) {
    const buf = await fetchCachedRange(url, 0, size, onProgress);
    try {
      const parsed = parseGguf(new Uint8Array(buf));
      if (parsed.dataOffset <= buf.byteLength) return parsed;
      lastError = new Error(`GGUF header extends past ${size} bytes`);
    } catch (e) {
      lastError = e;
    }
    size *= 2;
  }
  throw new Error(`could not parse GGUF header for ${url}: ${String(lastError)}`);
}

export function parseGguf(bytes: Uint8Array): GgufFile {
  const reader = new BinaryReader(bytes);
  const magic = reader.u32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(`invalid GGUF magic: 0x${magic.toString(16)}`);
  }
  const version = reader.u32();
  const tensorCount = reader.u64n();
  const kvCount = reader.u64n();
  const kv = new Map<string, GgufValue>();

  for (let i = 0; i < kvCount; i++) {
    const key = reader.string();
    const type = reader.u32();
    kv.set(key, reader.value(type));
  }

  const tensors: GgufTensorInfo[] = [];
  const tensorMap = new Map<string, GgufTensorInfo>();
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.string();
    const nDims = reader.u32();
    const shape: number[] = [];
    for (let d = 0; d < nDims; d++) shape.push(reader.u64n());
    const type = reader.u32();
    const offset = reader.u64n();
    const nElements = shape.reduce((a, b) => a * b, 1);
    const nBytes = ggmlTensorByteLength(type, nElements);
    const info = { name, shape, type, offset, nElements, nBytes };
    tensors.push(info);
    tensorMap.set(name, info);
  }

  return {
    version,
    tensorCount,
    kvCount,
    kv,
    tensors,
    tensorMap,
    dataOffset: align(reader.offset, 32),
  };
}

export function ggmlTensorByteLength(type: number, nElements: number): number {
  switch (type) {
    case GGML_TYPE.F32:
      return nElements * 4;
    case GGML_TYPE.F16:
      return nElements * 2;
    case GGML_TYPE.Q5_0:
      assertBlockMultiple(type, nElements, 32);
      return (nElements / 32) * 22;
    case GGML_TYPE.Q5_1:
      assertBlockMultiple(type, nElements, 32);
      return (nElements / 32) * 24;
    case GGML_TYPE.Q8_0:
      assertBlockMultiple(type, nElements, 32);
      return (nElements / 32) * 34;
    case GGML_TYPE.Q4_K:
      assertBlockMultiple(type, nElements, 256);
      return (nElements / 256) * 144;
    case GGML_TYPE.Q5_K:
      assertBlockMultiple(type, nElements, 256);
      return (nElements / 256) * 176;
    case GGML_TYPE.Q6_K:
      assertBlockMultiple(type, nElements, 256);
      return (nElements / 256) * 210;
    default:
      throw new Error(`unsupported GGML tensor type ${type}`);
  }
}

export function ggmlTypeName(type: number): string {
  switch (type) {
    case GGML_TYPE.F32:
      return "F32";
    case GGML_TYPE.F16:
      return "F16";
    case GGML_TYPE.Q5_0:
      return "Q5_0";
    case GGML_TYPE.Q5_1:
      return "Q5_1";
    case GGML_TYPE.Q8_0:
      return "Q8_0";
    case GGML_TYPE.Q4_K:
      return "Q4_K";
    case GGML_TYPE.Q5_K:
      return "Q5_K";
    case GGML_TYPE.Q6_K:
      return "Q6_K";
    default:
      return `type-${type}`;
  }
}

function assertBlockMultiple(type: number, nElements: number, block: number): void {
  if (nElements % block !== 0) {
    throw new Error(`GGML tensor type ${type} requires ${block}-element blocks, got ${nElements}`);
  }
}

function align(n: number, multiple: number): number {
  return Math.ceil(n / multiple) * multiple;
}

class BinaryReader {
  private readonly view: DataView;
  private readonly decoder = new TextDecoder("utf-8");
  offset = 0;

  constructor(private readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  u8(): number {
    const v = this.view.getUint8(this.offset);
    this.offset += 1;
    return v;
  }

  i8(): number {
    const v = this.view.getInt8(this.offset);
    this.offset += 1;
    return v;
  }

  u16(): number {
    const v = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return v;
  }

  i16(): number {
    const v = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return v;
  }

  u32(): number {
    const v = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return v;
  }

  i32(): number {
    const v = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return v;
  }

  u64(): bigint {
    const v = this.view.getBigUint64(this.offset, true);
    this.offset += 8;
    return v;
  }

  i64(): bigint {
    const v = this.view.getBigInt64(this.offset, true);
    this.offset += 8;
    return v;
  }

  u64n(): number {
    const v = this.u64();
    const n = Number(v);
    if (!Number.isSafeInteger(n)) throw new Error(`GGUF uint64 exceeds JS safe integer: ${v}`);
    return n;
  }

  f32(): number {
    const v = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return v;
  }

  f64(): number {
    const v = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return v;
  }

  string(): string {
    const length = this.u64n();
    const start = this.offset;
    this.offset += length;
    return this.decoder.decode(this.bytes.subarray(start, start + length));
  }

  value(type: number): GgufValue {
    switch (type) {
      case 0:
        return { type: "uint8", value: this.u8() };
      case 1:
        return { type: "int8", value: this.i8() };
      case 2:
        return { type: "uint16", value: this.u16() };
      case 3:
        return { type: "int16", value: this.i16() };
      case 4:
        return { type: "uint32", value: this.u32() };
      case 5:
        return { type: "int32", value: this.i32() };
      case 6:
        return { type: "float32", value: this.f32() };
      case 7:
        return { type: "bool", value: this.u8() !== 0 };
      case 8:
        return { type: "string", value: this.string() };
      case 9: {
        const itemType = this.u32();
        const length = this.u64n();
        const value: GgufValue[] = [];
        for (let i = 0; i < length; i++) value.push(this.value(itemType));
        return { type: "array", value };
      }
      case 10:
        return { type: "uint64", value: this.u64() };
      case 11:
        return { type: "int64", value: this.i64() };
      case 12:
        return { type: "float64", value: this.f64() };
      default:
        throw new Error(`unsupported GGUF metadata value type ${type}`);
    }
  }
}
