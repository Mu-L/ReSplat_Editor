import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import ts from 'typescript';

const sourceUrl = new URL('../src/paint-diagnostic.ts', import.meta.url);
const source = readFileSync(sourceUrl, 'utf8');
const compiled = ts.transpileModule(source, {
    compilerOptions: {
        module: ts.ModuleKind.ES2022,
        target: ts.ScriptTarget.ES2022
    }
}).outputText;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`;
const {
    PaintDiagnosticSampleCollector,
    buildPaintDiagnostic,
    classifyPaintSources,
    selectEvenlySpacedIds,
    summarizeNumbers
} = await import(moduleUrl);

const classification = classifyPaintSources(
    new Uint32Array([1, 2, 3, 4]),
    new Uint32Array([1, 3]),
    new Uint32Array([2, 3])
);
assert.deepEqual([...classification.sphereOnly], [1]);
assert.deepEqual([...classification.paintThroughOnly], [2]);
assert.deepEqual([...classification.overlap], [3]);
assert.deepEqual([...classification.unclassified], [4]);
assert.equal(classification.frontVisibleSupplement, 0);
assert.equal(classification.frontVisibleSupplementOnly, 0);
const supplementedClassification = classifyPaintSources(
    new Uint32Array([1, 2, 3]),
    new Uint32Array([1]),
    new Uint32Array([]),
    new Uint32Array([1, 2])
);
assert.deepEqual([...supplementedClassification.sphereOnly], [1, 2]);
assert.deepEqual([...supplementedClassification.unclassified], [3]);
assert.equal(supplementedClassification.frontVisibleSupplement, 2);
assert.equal(supplementedClassification.frontVisibleSupplementOnly, 1);
assert.deepEqual(selectEvenlySpacedIds(new Uint32Array([0, 1, 2, 3, 4]), 3), [0, 2, 4]);
assert.deepEqual(summarizeNumbers([4, 1, 3, 2]), {
    count: 4,
    totalMs: 10,
    minMs: 1,
    p50Ms: 2,
    p95Ms: 3,
    maxMs: 4
});

const sample = (sequence) => ({
    sequence,
    screen: { x: sequence / 1000, y: 0.5 },
    enqueuedAtMs: sequence,
    startedAtMs: sequence,
    completedAtMs: sequence + 1,
    outcome: 'painted',
    surface: {
        kind: 'target',
        worldPosition: [0, 0, 0],
        modelPosition: [0, 0, 0],
        distanceFromCamera: 1
    },
    brush: {
        radiusPixels: 10,
        modelRadius: 0.2,
        interpolationSteps: 1,
        frontSurfaceDepthToleranceWorldUnits: 0.02
    },
    paintThrough: {
        id: sequence,
        queued: true,
        depthBehindSurface: sequence === 999 ? 0.5 : -0.1,
        depthBehindSurfaceInBrushRadii: sequence === 999 ? 2.5 : -0.5
    },
    timings: { depthPickMs: 1, idReadbackMs: 2, processingMs: 4 }
});

const collector = new PaintDiagnosticSampleCollector(500);
for (let index = 0; index < 1000; index++) collector.add(sample(index));
const retained = collector.snapshot();
assert.equal(retained.total, 1000);
assert.ok(retained.retained.length <= 500);
assert.equal(retained.retained[0].sequence, 0);
assert.equal(retained.retained.at(-1).sequence, 999);
assert.ok(retained.omitted > 0);

const context = {
    splat: { filename: 'example.ply', name: 'Example', gaussianCount: 10 },
    layer: { id: 'layer-1', name: 'Paint 1', visible: true, opacity: 1, blendMode: 'normal' },
    brush: {
        color: [1, 0, 0, 1],
        strength: 0.5,
        hardness: 1,
        radiusPixels: 10,
        selectionMode: 'screenCircleFrontSurface',
        frontSurfaceDepthToleranceRatio: 0.1
    },
    picking: { surfaceAlphaThreshold: 0.2, paintThroughAlphaThreshold: 1 / 255, paintThroughEnabled: false },
    camera: {
        position: [0, 0, -1], forward: [0, 0, 1], focalPoint: [0, 0, 0],
        fovDegrees: 60, orthographic: false, nearClip: 0.01, farClip: 1000
    },
    viewport: {
        cssWidth: 800, cssHeight: 600, renderWidth: 800, renderHeight: 600,
        devicePixelRatio: 1, graphicsBackend: 'webgl2'
    },
    target: {
        worldTransformColumnMajor: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        localBounds: { center: [0, 0, 0], halfExtents: [1, 1, 1] },
        worldBounds: { center: [0, 0, 0], halfExtents: [1, 1, 1] },
        gaussianStates: {
            total: 10, selected: 0, locked: 1, deleted: 0, soloHidden: 0,
            independentlyEdited: 0, spherePaintEligible: 9, idPickEligible: 9
        }
    }
};

const report = buildPaintDiagnostic({
    createdAt: '2026-09-12T00:00:00.000Z',
    completedAt: '2026-09-12T00:00:01.000Z',
    status: 'success',
    context,
    inputEvents: 12,
    coalescedInputEvents: 6,
    samples: [sample(0), sample(999)],
    totalSamples: 2,
    sampleTotals: {
        target: 2, miss: 0, otherSplat: 0,
        paintThroughBehindSurface: 1, maximumPaintThroughDepthBehindSurface: 0.5
    },
    sourceClassification: classification,
    representativeGaussians: { sphereOnly: [], paintThroughOnly: [], overlap: [], unclassified: [] },
    performance: { queueDrainMs: 8, commitMs: 10, gpuReadbackMs: 6, commitCpuMs: 4, totalStrokeMs: 30 }
});

assert.equal(report.schema, 'resplat-paint-brush-diagnostic');
assert.equal(report.version, 1);
assert.equal(report.metrics.paintSources.sphereOnly, 1);
assert.equal(report.context.brush.selectionMode, 'screenCircleFrontSurface');
assert.equal(report.metrics.frontSurfaceDepthToleranceWorldUnits.p50, 0.02);
assert.equal(report.analysisGuide.some(line => line.includes('screen-space circle plus front-surface depth filter')), true);
assert.equal(report.analysisGuide.some(line => line.includes('effectiveStrength')), true);
assert.equal(report.analysisGuide.some(line => line.includes('frontVisibleFootprintSupplementEnabled')), true);
assert.equal(report.analysisGuide.some(line => line.includes('spherical harmonics')), true);
assert.equal(report.metrics.modelRadius.max, 0.2);
assert.ok(report.chatgptSummary.every(value => typeof value === 'string' && value.length > 0));
assert.ok(report.signals.some(value => value.code === 'PAINT_THROUGH_BEHIND_SURFACE'));
assert.ok(report.signals.some(value => value.code === 'INPUT_EVENTS_COALESCED'));
assert.ok(report.signals.some(value => value.code === 'TARGET_CONTAINS_FILTERED_GAUSSIANS'));
assert.equal(JSON.stringify(report).includes('base64'), false);

for (const [status, expected] of [
    ['empty', 'ZERO_CHANGED_GAUSSIANS'],
    ['cancelled', 'STROKE_CANCELLED'],
    ['error', 'STROKE_ERROR']
]) {
    const emptyClassification = classifyPaintSources([], [], []);
    const variant = buildPaintDiagnostic({
        createdAt: '2026-09-12T00:00:00.000Z',
        completedAt: '2026-09-12T00:00:01.000Z',
        status,
        reason: status === 'cancelled' ? 'pointer-cancelled' : undefined,
        error: status === 'error' ? 'readback failed' : undefined,
        context,
        inputEvents: 1,
        coalescedInputEvents: 0,
        samples: [],
        totalSamples: 0,
        sampleTotals: {
            target: 0, miss: 0, otherSplat: 0,
            paintThroughBehindSurface: 0, maximumPaintThroughDepthBehindSurface: 0
        },
        sourceClassification: emptyClassification,
        representativeGaussians: { sphereOnly: [], paintThroughOnly: [], overlap: [], unclassified: [] },
        performance: { queueDrainMs: 0, commitMs: 0, gpuReadbackMs: 0, commitCpuMs: 0, totalStrokeMs: 1 }
    });
    assert.equal(variant.status.value, status);
    assert.ok(variant.signals.some(value => value.code === expected));
}

console.log('paint diagnostic report tests passed');
