import { describe, expect, it } from 'vitest';

import {
    applyRefineMask,
    carveVisibleOutside,
    chooseStablePromptSample,
    evaluateAuxiliaryViewSafety,
    evaluateSegmentMaskQuality,
    fuseSegmentObservations,
    selectSafestAuxiliaryPose,
    suppressGroundPlaneLeak,
    collectVisibilitySupport,
    filterAnchorConnectedSelection,
    findMaskInteriorPoint,
    hasMaskHits,
    projectMaskToSource,
    projectPredictionToSource,
    resolveCarvedSelection,
    sampleDistributedMaskPoints,
    sampleMaskPoints,
    selectBestMask
} from './mask-utils';

describe('SAM2 mask helpers', () => {
    it('moves an isolated center hit to the nearest stable surface in the 9x9 neighborhood', () => {
        const invalid = 0xffffffff;
        const low = new Uint32Array(81).fill(invalid);
        const high = new Uint32Array(81).fill(invalid);
        low[4 * 9 + 4] = 0;
        for (const [x, y] of [[5, 3], [6, 3], [5, 4], [6, 4], [5, 5], [6, 5]]) {
            low[y * 9 + x] = 1;
            high[y * 9 + x] = 1;
        }

        expect(chooseStablePromptSample(low, high, 9, 9, {
            centers: new Float32Array([
                20, 20, 20,
                0, 0, 0,
                0.1, 0, 0,
                0, 0.1, 0,
                0, 0, 0.1,
                -0.1, 0, 0,
                0, -0.1, 0,
                0, 0, -0.1
            ]),
            medianScale: 0.1
        })).toEqual({ id: 1, x: 5, y: 4, stable: true });
    });

    it('rejects an auxiliary view whose first visible surface is a wall far from the anchor', () => {
        expect(evaluateAuxiliaryViewSafety({
            anchorInViewport: true,
            cameraAnchorDistance: 10,
            firstHitDistance: 1,
            firstHitAnchorDistance: 9,
            targetRadius: 1,
            medianScale: 0.01
        })).toEqual({ accepted: false, reason: 'anchor-mismatch' });
    });

    it('chooses the safe reroute with the greatest camera clearance', () => {
        expect(selectSafestAuxiliaryPose([
            { value: 'yaw-left', safety: { accepted: false, reason: 'anchor-mismatch' }, clearance: 5 },
            { value: 'pitch-up', safety: { accepted: true }, clearance: 3 },
            { value: 'pitch-down', safety: { accepted: true }, clearance: 1 }
        ])).toBe('pitch-up');
    });

    it('rejects a wall-like auxiliary mask that covers most of the viewport', () => {
        expect(evaluateSegmentMaskQuality({
            predictedIou: 0.95,
            maskPixels: 7500,
            totalPixels: 10000,
            promptInside: true
        })).toEqual({ accepted: false, reason: 'oversized-mask' });
    });

    it('rejects a high-IoU auxiliary mask that is tiny relative to the current target', () => {
        expect(evaluateSegmentMaskQuality({
            predictedIou: 0.95,
            maskPixels: 300,
            totalPixels: 10000,
            promptInside: true,
            referenceMaskPixels: 8000,
            minimumRelativeArea: 0.08
        })).toEqual({ accepted: false, reason: 'undersized-mask' });
    });

    it('rejects a partial auxiliary mask that covers only one sixth of the current target', () => {
        expect(evaluateSegmentMaskQuality({
            predictedIou: 0.95,
            maskPixels: 1600,
            totalPixels: 10000,
            promptInside: true,
            referenceMaskPixels: 10000
        })).toEqual({ accepted: false, reason: 'undersized-mask' });
    });

    it('selects the candidate with the highest predicted IoU', () => {
        const logits = new Float32Array([
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3
        ]);
        const selected = selectBestMask(logits, [1, 3, 2, 2], new Float32Array([0.1, 0.9, 0.2]));
        expect(selected.candidate).toBe(1);
        expect(Array.from(selected.logits)).toEqual([2, 2, 2, 2]);
    });

    it('prefers a target-sized candidate over a higher-IoU fragment', () => {
        const logits = new Float32Array([
            1, -1, -1, -1,
            1, 1, 1, -1
        ]);
        const selected = selectBestMask(
            logits,
            [1, 2, 2, 2],
            new Float32Array([0.95, 0.8]),
            { referenceMaskRatio: 0.75, minimumRelativeArea: 0.5 }
        );
        expect(selected.candidate).toBe(1);
    });

    it('projects logits through the inverse letterbox mapping', () => {
        const projected = projectMaskToSource({
            logits: new Float32Array([
                -1, -1, -1, -1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                -1, -1, -1, -1
            ]),
            width: 4,
            height: 4,
            candidate: 0
        }, {
            sourceWidth: 2,
            sourceHeight: 1,
            modelSize: 4,
            scaledWidth: 4,
            scaledHeight: 2,
            scaleX: 2,
            scaleY: 2,
            padX: 0,
            padY: 1
        });
        expect(Array.from(projected)).toEqual([255, 255]);
    });

    it('bilinearly interpolates logits before thresholding', () => {
        const projected = projectMaskToSource({
            logits: new Float32Array([-1, 1]),
            width: 2,
            height: 1,
            candidate: 0
        }, {
            sourceWidth: 3,
            sourceHeight: 1,
            modelSize: 3,
            scaledWidth: 3,
            scaledHeight: 3,
            scaleX: 1,
            scaleY: 3,
            padX: 0,
            padY: 0
        });
        expect(Array.from(projected)).toEqual([0, 0, 255]);
    });

    it('keeps bilinearly projected SAM logits as quantized confidence', () => {
        const prediction = projectPredictionToSource({
            logits: new Float32Array([0, Math.log(3)]),
            width: 2,
            height: 1,
            candidate: 0,
            predictedIou: 0.8
        }, {
            sourceWidth: 2,
            sourceHeight: 1,
            modelSize: 2,
            scaledWidth: 2,
            scaledHeight: 1,
            scaleX: 1,
            scaleY: 1,
            padX: 0,
            padY: 0
        });
        expect(Array.from(prediction.mask)).toEqual([0, 255]);
        expect(Array.from(prediction.confidence)).toEqual([128, 191]);
        expect(prediction.predictedIou).toBeCloseTo(0.8);
    });

    it('uses continuous multi-view evidence to keep the target and reject visible ground', () => {
        const fused = fuseSegmentObservations(new Uint8Array([255, 255, 255, 255]), [{
            inside: new Uint8Array([255, 0, 255, 0]),
            visible: new Uint8Array([255, 0, 255, 0]),
            confidence: new Uint8Array([230, 0, 150, 0]),
            predictedIou: 0.9
        }, {
            inside: new Uint8Array([255, 255, 0, 0]),
            visible: new Uint8Array([255, 255, 255, 0]),
            confidence: new Uint8Array([220, 210, 20, 0]),
            predictedIou: 0.85
        }]);

        expect(Array.from(fused.candidates)).toEqual([255, 255, 0, 255]);
        expect(Array.from(fused.support)).toEqual([255, 255, 0, 0]);
        expect(Array.from(fused.visibleOutside)).toEqual([0, 0, 255, 0]);
    });

    it('confirms hidden center candidates only when two auxiliary silhouettes agree', () => {
        const empty = new Uint8Array(4);
        const fused = fuseSegmentObservations(new Uint8Array([255, 255, 255, 255]), [{
            inside: empty,
            visible: empty,
            silhouette: new Uint8Array([255, 255, 255, 0])
        }, {
            inside: empty,
            visible: empty,
            silhouette: new Uint8Array([255, 255, 0, 255])
        }]);

        expect(Array.from(fused.silhouetteCore)).toEqual([255, 255, 0, 0]);
    });

    it('removes low-confidence single-view ground outside the trusted target footprint', () => {
        const count = 67;
        const centers = new Float32Array(count * 3);
        centers.set([0, 1, 0, 0.2, 0, 0, 2, 0, 0]);
        for (let index = 3; index < count; index++) {
            const sample = index - 3;
            centers[index * 3] = (sample % 8 - 3.5) * 0.5;
            centers[index * 3 + 1] = 0;
            centers[index * 3 + 2] = (Math.floor(sample / 8) - 3.5) * 0.5;
        }
        const visible = new Uint8Array(count);
        visible.fill(255, 3);
        const hits = new Uint8Array(count);
        hits[0] = hits[1] = hits[2] = 255;
        const support = new Uint8Array(count);
        support[0] = support[1] = 255;
        const positiveViews = new Uint8Array(count);
        positiveViews[0] = 2;
        positiveViews[1] = 1;
        positiveViews[2] = 1;
        const maxConfidence = new Uint8Array(count);
        maxConfidence[0] = 230;
        maxConfidence[1] = 230;
        maxConfidence[2] = 150;

        const result = suppressGroundPlaneLeak(hits, support, positiveViews, maxConfidence, visible, {
            centers,
            anchor: { x: 0, y: 1, z: 0 },
            up: { x: 0, y: 1, z: 0 },
            targetRadius: 2,
            medianScale: 0.05
        });
        expect(result.detected).toBe(true);
        expect(Array.from(result.hits.slice(0, 3))).toEqual([255, 255, 0]);
    });

    it('finds a point inside the clicked mask component away from its boundary', () => {
        const mask = new Uint8Array(25);
        for (let y = 1; y <= 3; y++) {
            for (let x = 1; x <= 3; x++) mask[y * 5 + x] = 255;
        }
        expect(findMaskInteriorPoint(mask, 5, 5, { x: 0.2, y: 0.2 })).toEqual({ x: 0.5, y: 0.5 });
        expect(sampleMaskPoints(mask, 5, 5, { x: 0.5, y: 0.5 }, 4)[0]).toEqual({ x: 0.5, y: 0.5 });
    });

    it('spreads auxiliary prompts across the mask interior', () => {
        const mask = new Uint8Array(40 * 20);
        for (let y = 2; y < 18; y++) {
            for (let x = 4; x < 26; x++) mask[y * 40 + x] = 255;
        }
        for (let y = 2; y < 18; y++) {
            for (let x = 32; x < 38; x++) mask[y * 40 + x] = 255;
        }
        const points = sampleDistributedMaskPoints(mask, 40, 20, { x: 0.5, y: 0.5 }, 5);
        expect(points).toHaveLength(5);
        expect(Math.max(...points.map(point => point.x))).toBeLessThan(0.7);
        expect(Math.max(...points.map(point => point.x)) - Math.min(...points.map(point => point.x))).toBeGreaterThan(0.35);
        expect(Math.max(...points.map(point => point.y)) - Math.min(...points.map(point => point.y))).toBeGreaterThan(0.5);
    });

    it('carves only candidates observed outside a side-view mask', () => {
        const result = carveVisibleOutside(new Uint8Array([255, 255, 255, 255]), [{
            inside: new Uint8Array([255, 0, 255, 0]),
            visible: new Uint8Array([255, 255, 0, 0])
        }]);
        expect(Array.from(result)).toEqual([255, 0, 255, 255]);
    });

    it('uses current visible hits when no side view can contribute', () => {
        const result = resolveCarvedSelection(
            new Uint8Array([255, 255, 255]),
            new Uint8Array([255, 0, 0]),
            []
        );
        expect(Array.from(result.hits)).toEqual([255, 0, 0]);
        expect(result.degraded).toBe(true);
    });

    it('keeps a one-sided carved result but marks it degraded', () => {
        const result = resolveCarvedSelection(
            new Uint8Array([255, 255, 255]),
            new Uint8Array([255, 0, 0]),
            [{
                inside: new Uint8Array([255, 0, 0]),
                visible: new Uint8Array([255, 255, 0])
            }]
        );
        expect(Array.from(result.hits)).toEqual([255, 0, 255]);
        expect(result.degraded).toBe(true);
    });

    it('requires every configured auxiliary view before clearing degraded state', () => {
        const observations = [{
            inside: new Uint8Array([255, 255]),
            visible: new Uint8Array([255, 255])
        }, {
            inside: new Uint8Array([255, 255]),
            visible: new Uint8Array([255, 255])
        }];
        const result = resolveCarvedSelection(
            new Uint8Array([255, 255]),
            new Uint8Array([255, 0]),
            observations,
            3
        );
        expect(Array.from(result.hits)).toEqual([255, 255]);
        expect(result.degraded).toBe(true);
    });

    it('falls back to current visible hits when carving removes every candidate', () => {
        const result = resolveCarvedSelection(
            new Uint8Array([255, 255]),
            new Uint8Array([0, 255]),
            [{
                inside: new Uint8Array([0, 0]),
                visible: new Uint8Array([255, 255])
            }]
        );
        expect(Array.from(result.hits)).toEqual([0, 255]);
        expect(result.degraded).toBe(true);
    });

    it('collects positively visible surfaces from the current and side views', () => {
        const result = collectVisibilitySupport(new Uint8Array([255, 0, 0, 0]), [{
            inside: new Uint8Array([0, 255, 0, 0]),
            visible: new Uint8Array([255, 255, 255, 0])
        }, {
            inside: new Uint8Array([0, 0, 255, 0]),
            visible: new Uint8Array([0, 255, 255, 255])
        }]);
        expect(Array.from(result)).toEqual([255, 255, 255, 0]);
    });

    it('keeps the anchor-connected target and removes a separated background layer', () => {
        const centers = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            10, 0, 0,
            11, 0, 0
        ]);
        const scales = new Float32Array(6).fill(Math.log(0.5));
        const result = filterAnchorConnectedSelection(
            new Uint8Array([255, 255, 255, 255, 255, 255]),
            new Uint8Array([255, 0, 0, 255, 0, 0]),
            { centers, scales: [scales, scales, scales] },
            { x: 0, y: 0, z: 0 }
        );
        expect(Array.from(result.hits)).toEqual([255, 255, 255, 255, 0, 0]);
        expect(result.degraded).toBe(false);
    });

    it('uses color discontinuity to stop growth into touching scenery', () => {
        const centers = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            5, 0, 0
        ]);
        const scales = new Float32Array(6).fill(Math.log(0.5));
        const red = new Float32Array([0, 0, 0, 0, 3, 3]);
        const green = new Float32Array([0, 0, 0, 0, 3, 3]);
        const blue = new Float32Array([0, 0, 0, 0, 3, 3]);
        const result = filterAnchorConnectedSelection(
            new Uint8Array([255, 255, 255, 255, 255, 255]),
            new Uint8Array([255, 0, 0, 255, 0, 0]),
            { centers, scales: [scales, scales, scales], colors: [red, green, blue] },
            { x: 0, y: 0, z: 0 }
        );
        expect(Array.from(result.hits)).toEqual([255, 255, 255, 255, 0, 0]);
    });

    it('limits same-color propagation to a short halo around visible support', () => {
        const centers = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            5, 0, 0
        ]);
        const scales = new Float32Array(6).fill(Math.log(0.5));
        const result = filterAnchorConnectedSelection(
            new Uint8Array([255, 255, 255, 255, 255, 255]),
            new Uint8Array([255, 0, 0, 0, 0, 0]),
            { centers, scales: [scales, scales, scales] },
            { x: 0, y: 0, z: 0 }
        );
        expect(Array.from(result.hits)).toEqual([255, 255, 255, 255, 0, 0]);
        expect(result.degraded).toBe(false);
    });

    it('keeps hidden target volume confirmed by multiple silhouettes beyond the three-hop surface halo', () => {
        const centers = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            5, 0, 0,
            6, 0, 0,
            7, 0, 0
        ]);
        const scales = new Float32Array(8).fill(Math.log(0.5));
        const result = filterAnchorConnectedSelection(
            new Uint8Array(8).fill(255),
            new Uint8Array([255, 0, 0, 0, 0, 0, 0, 0]),
            { centers, scales: [scales, scales, scales] },
            { x: 0, y: 0, z: 0 },
            new Uint8Array(8).fill(255)
        );
        expect(Array.from(result.hits)).toEqual([255, 255, 255, 255, 255, 255, 255, 255]);
        expect(result.degraded).toBe(false);
    });

    it('rejects a disconnected background layer even when its centers share the target silhouettes', () => {
        const centers = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            5, 0, 0,
            20, 0, 0,
            21, 0, 0
        ]);
        const scales = new Float32Array(8).fill(Math.log(0.5));
        const result = filterAnchorConnectedSelection(
            new Uint8Array(8).fill(255),
            new Uint8Array([255, 0, 0, 0, 0, 0, 0, 0]),
            { centers, scales: [scales, scales, scales] },
            { x: 0, y: 0, z: 0 },
            new Uint8Array(8).fill(255)
        );
        expect(Array.from(result.hits)).toEqual([255, 255, 255, 255, 255, 255, 0, 0]);
        expect(result.degraded).toBe(false);
    });

    it('falls back to positively observed surfaces when local growth explodes', () => {
        const centers = new Float32Array([
            0, 0, 0,
            0.1, 0, 0,
            0.2, 0, 0,
            0.3, 0, 0,
            0.4, 0, 0,
            0.5, 0, 0,
            0.6, 0, 0
        ]);
        const scales = new Float32Array(7).fill(Math.log(0.5));
        const result = filterAnchorConnectedSelection(
            new Uint8Array(7).fill(255),
            new Uint8Array([255, 0, 0, 0, 0, 0, 0]),
            { centers, scales: [scales, scales, scales] },
            { x: 0, y: 0, z: 0 }
        );
        expect(Array.from(result.hits)).toEqual([255, 0, 0, 0, 0, 0, 0]);
        expect(result.degraded).toBe(true);
    });

    it('refines only currently selected Gaussians', () => {
        const result = applyRefineMask(
            new Uint8Array([255, 255, 255, 0]),
            new Uint8Array([1, 0, 3, 1]),
            1
        );
        expect(Array.from(result)).toEqual([255, 0, 255, 0]);
    });

    it('rejects visibility masks with different lengths', () => {
        expect(() => carveVisibleOutside(new Uint8Array(1), [{
            inside: new Uint8Array(1),
            visible: new Uint8Array(2)
        }])).toThrow('different sizes');
    });
});
