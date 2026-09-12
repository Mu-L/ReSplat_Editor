// The pinned vietanhdev encoder declares two intermediate Conv outputs with
// an empty (scalar) shape. Native ONNX Runtime accepts those declarations with
// a lenient merge, but ONNX Runtime Web rejects them before creating a session.
//
// The model checksum is verified before this runs. Replacing the empty `shape`
// protobuf tag (field 2) with an unknown tag removes only the invalid shape
// declaration while preserving the byte length and the inferred tensor shape.
const INVALID_VALUE_INFO_NAMES = [
    '/conv_s0/Conv_output_0',
    '/conv_s1/Conv_output_0'
] as const;

const INVALID_TENSOR_TYPE_SUFFIX = new Uint8Array([
    0x12, 0x06, // ValueInfoProto.type, 6 bytes
    0x0a, 0x04, // TypeProto.tensor_type, 4 bytes
    0x08, 0x01, // Tensor.elem_type = FLOAT
    0x12, 0x00  // Tensor.shape = an invalid empty shape
]);
const VALUE_INFO_SEARCH_TAIL_BYTES = 1024 * 1024;

const hasBytesAt = (bytes: Uint8Array, expected: Uint8Array, offset: number) => {
    if (offset < 0 || offset + expected.length > bytes.length) return false;
    for (let index = 0; index < expected.length; index++) {
        if (bytes[offset + index] !== expected[index]) return false;
    }
    return true;
};

const patchSam2EncoderForOrtWeb = (bytes: Uint8Array) => {
    const encoder = new TextEncoder();
    for (const valueInfoName of INVALID_VALUE_INFO_NAMES) {
        const name = encoder.encode(valueInfoName);
        let patched = false;

        // Value-info records sit near the end of this pinned model, so search
        // backwards and ignore the same names used earlier by graph nodes.
        const firstOffset = Math.max(0, bytes.length - VALUE_INFO_SEARCH_TAIL_BYTES);
        for (let offset = bytes.length - name.length - INVALID_TENSOR_TYPE_SUFFIX.length; offset >= firstOffset; offset--) {
            if (!hasBytesAt(bytes, name, offset)) continue;
            const suffixOffset = offset + name.length;
            if (!hasBytesAt(bytes, INVALID_TENSOR_TYPE_SUFFIX, suffixOffset)) continue;

            bytes[suffixOffset + INVALID_TENSOR_TYPE_SUFFIX.length - 2] = 0x7a;
            patched = true;
            break;
        }

        if (!patched) throw new Error(`Unexpected SAM2 encoder metadata for ${valueInfoName}`);
    }
    return bytes;
};

export { patchSam2EncoderForOrtWeb };
