import { describe, expect, it } from 'vitest';

import { patchSam2EncoderForOrtWeb } from './sam2-model';

const invalidValueInfo = (name: string) => [
    ...new TextEncoder().encode(name),
    0x12, 0x06, 0x0a, 0x04, 0x08, 0x01, 0x12, 0x00
];

describe('SAM2 ONNX compatibility patch', () => {
    it('removes the two invalid scalar shape declarations in place', () => {
        const bytes = new Uint8Array([
            0x01,
            ...invalidValueInfo('/conv_s0/Conv_output_0'),
            0x02,
            ...invalidValueInfo('/conv_s1/Conv_output_0'),
            0x03
        ]);

        expect(patchSam2EncoderForOrtWeb(bytes)).toBe(bytes);
        expect(Array.from(bytes).filter(value => value === 0x7a)).toHaveLength(2);
    });

    it('rejects model bytes that do not match the pinned metadata contract', () => {
        expect(() => patchSam2EncoderForOrtWeb(new Uint8Array([0x01]))).toThrow('Unexpected SAM2 encoder metadata');
    });
});
