import { describe, expect, it } from 'vitest';

import { normalizedPointToModel, normalizedPointsToModel, prepareSam2Image, sam2FrameSignature } from './sam2-image';

describe('SAM2 image preprocessing', () => {
    it('resizes the source directly to the square encoder input', () => {
        const rgba = new Uint8Array([
            0, 0, 0, 255,
            255, 255, 255, 255
        ]);
        const prepared = prepareSam2Image(rgba, 2, 1, 4);

        expect(prepared.transform).toMatchObject({
            scaledWidth: 4,
            scaledHeight: 4,
            padX: 0,
            padY: 0,
            scaleX: 2,
            scaleY: 4
        });
        expect(prepared.tensorData).toHaveLength(4 * 4 * 3);
    });

    it('maps normalized clicks through independent x/y scales', () => {
        const prepared = prepareSam2Image(new Uint8Array(8), 2, 1, 4);
        expect(normalizedPointToModel({ x: 0.5, y: 0.5 }, prepared.transform)).toEqual([2, 2]);
        expect(Array.from(normalizedPointsToModel([
            { x: 0.25, y: 0.75 },
            { x: 1, y: 0 }
        ], prepared.transform))).toEqual([1, 3, 4, 0]);
    });

    it('changes the frame signature when any RGB pixel changes', () => {
        const first = new Uint8Array(4 * 4 * 4);
        const second = new Uint8Array(first);
        second[second.length - 6] = 255;
        expect(sam2FrameSignature(first, 4, 4)).not.toBe(sam2FrameSignature(second, 4, 4));
    });

    it('rejects inconsistent image dimensions', () => {
        expect(() => prepareSam2Image(new Uint8Array(3), 1, 1)).toThrow('Invalid RGBA image dimensions');
    });
});
