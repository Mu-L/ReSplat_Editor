type DiagnosticPoint = { x: number; y: number };
type DiagnosticVec3 = [number, number, number];
type PaintDiagnosticStatus = 'success' | 'empty' | 'cancelled' | 'error';
type PaintDiagnosticSource = 'sphereOnly' | 'paintThroughOnly' | 'overlap' | 'unclassified';

type PaintDiagnosticGaussian = {
    id: number;
    source: PaintDiagnosticSource | 'paintThroughSample';
    modelPosition: DiagnosticVec3;
    worldPosition: DiagnosticVec3;
    cameraDepth: number;
    opacity: number | null;
    state: {
        raw: number;
        selected: boolean;
        locked: boolean;
        deleted: boolean;
    };
    soloVisible: boolean;
    independentlyEdited: boolean;
};

type PaintDiagnosticSample = {
    sequence: number;
    screen: DiagnosticPoint;
    enqueuedAtMs: number;
    startedAtMs: number;
    completedAtMs: number;
    outcome: 'painted' | 'miss' | 'other-splat';
    surface: {
        kind: 'target' | 'miss' | 'other-splat';
        splat?: { filename: string; name: string; gaussianCount: number };
        worldPosition?: DiagnosticVec3;
        modelPosition?: DiagnosticVec3;
        distanceFromCamera?: number;
    };
    brush?: {
        radiusPixels: number;
        modelRadius: number;
        interpolationSteps: number;
        frontSurfaceDepthToleranceWorldUnits?: number;
    };
    paintThrough?: {
        id: number | null;
        queued: boolean;
        gaussian?: PaintDiagnosticGaussian;
        // Positive means that the Gaussian center is farther along the camera
        // forward axis than the depth-picked surface point.
        depthBehindSurface?: number;
        depthBehindSurfaceInBrushRadii?: number;
    };
    timings: {
        depthPickMs: number;
        idReadbackMs: number;
        processingMs: number;
    };
};

type PaintDiagnosticSignal = {
    code: string;
    severity: 'info' | 'warning' | 'error';
    evidence: string;
    interpretation: string;
};

type PaintSourceClassification = {
    changedCount: number;
    frontVisibleSupplement: number;
    frontVisibleSupplementOnly: number;
    sphereOnly: Uint32Array;
    paintThroughOnly: Uint32Array;
    overlap: Uint32Array;
    unclassified: Uint32Array;
};

type PaintDiagnosticContext = {
    splat: {
        filename: string;
        name: string;
        gaussianCount: number;
        availableSphericalHarmonicBands?: number;
        activeSphericalHarmonicBands?: number;
    };
    layer: {
        id: string | null;
        name: string | null;
        visible: boolean;
        opacity: number | null;
        blendMode: string | null;
    };
    brush: {
        color: [number, number, number, number];
        strength: number;
        effectiveStrength?: number;
        hardness: number;
        radiusPixels: number;
        selectionMode?: 'modelSphere' | 'screenCircleFrontSurface';
        frontSurfaceDepthToleranceRatio?: number;
    };
    picking: {
        surfaceAlphaThreshold: number;
        paintThroughAlphaThreshold: number;
        paintThroughEnabled?: boolean;
        frontVisibleFootprintSupplementEnabled?: boolean;
    };
    camera: {
        position: DiagnosticVec3;
        forward: DiagnosticVec3;
        focalPoint: DiagnosticVec3;
        fovDegrees: number;
        orthographic: boolean;
        nearClip: number;
        farClip: number;
    };
    viewport: {
        cssWidth: number;
        cssHeight: number;
        renderWidth: number;
        renderHeight: number;
        devicePixelRatio: number;
        graphicsBackend: string;
    };
    target: {
        worldTransformColumnMajor: number[];
        localBounds: { center: DiagnosticVec3; halfExtents: DiagnosticVec3 };
        worldBounds: { center: DiagnosticVec3; halfExtents: DiagnosticVec3 };
        gaussianStates: {
            total: number;
            selected: number;
            locked: number;
            deleted: number;
            soloHidden: number;
            independentlyEdited: number;
            spherePaintEligible: number;
            idPickEligible: number;
        };
    };
};

type PaintDiagnosticBuildInput = {
    createdAt: string;
    completedAt: string;
    status: PaintDiagnosticStatus;
    reason?: string;
    error?: string;
    context: PaintDiagnosticContext;
    inputEvents: number;
    coalescedInputEvents: number;
    samples: PaintDiagnosticSample[];
    totalSamples: number;
    sampleTotals: {
        target: number;
        miss: number;
        otherSplat: number;
        paintThroughBehindSurface: number;
        maximumPaintThroughDepthBehindSurface: number;
    };
    sourceClassification: PaintSourceClassification;
    representativeGaussians: Record<PaintDiagnosticSource, PaintDiagnosticGaussian[]>;
    performance: {
        queueDrainMs: number;
        commitMs: number;
        gpuReadbackMs: number;
        commitCpuMs: number;
        totalStrokeMs: number;
    };
};

type TimingSummary = {
    count: number;
    totalMs: number;
    minMs: number;
    p50Ms: number;
    p95Ms: number;
    maxMs: number;
};

type DistributionSummary = {
    count: number;
    min: number;
    p50: number;
    p95: number;
    max: number;
};

const round = (value: number, digits = 4) => {
    if (!Number.isFinite(value)) return 0;
    const scale = 10 ** digits;
    return Math.round(value * scale) / scale;
};

const summarizeNumbers = (values: readonly number[]): TimingSummary | null => {
    const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (finite.length === 0) return null;
    const percentile = (fraction: number) => finite[Math.min(finite.length - 1, Math.floor((finite.length - 1) * fraction))];
    return {
        count: finite.length,
        totalMs: round(finite.reduce((sum, value) => sum + value, 0)),
        minMs: round(finite[0]),
        p50Ms: round(percentile(0.5)),
        p95Ms: round(percentile(0.95)),
        maxMs: round(finite[finite.length - 1])
    };
};

const summarizeDistribution = (values: readonly number[]): DistributionSummary | null => {
    const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (finite.length === 0) return null;
    const percentile = (fraction: number) => finite[Math.min(finite.length - 1, Math.floor((finite.length - 1) * fraction))];
    return {
        count: finite.length,
        min: round(finite[0]),
        p50: round(percentile(0.5)),
        p95: round(percentile(0.95)),
        max: round(finite[finite.length - 1])
    };
};

const classifyPaintSources = (
    changedIds: ArrayLike<number>,
    sphereIds: ArrayLike<number>,
    paintThroughIds: ArrayLike<number>,
    frontVisibleIds: ArrayLike<number> = []
): PaintSourceClassification => {
    const sphere = new Set<number>();
    const paintThrough = new Set<number>();
    const frontVisible = new Set<number>();
    for (let index = 0; index < sphereIds.length; index++) sphere.add(sphereIds[index]);
    for (let index = 0; index < paintThroughIds.length; index++) paintThrough.add(paintThroughIds[index]);
    for (let index = 0; index < frontVisibleIds.length; index++) frontVisible.add(frontVisibleIds[index]);

    const counts = [0, 0, 0, 0];
    let frontVisibleSupplement = 0;
    let frontVisibleSupplementOnly = 0;
    for (let index = 0; index < changedIds.length; index++) {
        const id = changedIds[index];
        const byFrontVisible = frontVisible.has(id);
        const bySphere = sphere.has(id) || byFrontVisible;
        const byPaintThrough = paintThrough.has(id);
        if (byFrontVisible) frontVisibleSupplement++;
        if (byFrontVisible && !sphere.has(id)) frontVisibleSupplementOnly++;
        counts[bySphere ? (byPaintThrough ? 2 : 0) : (byPaintThrough ? 1 : 3)]++;
    }

    const groups = counts.map(count => new Uint32Array(count));
    const offsets = [0, 0, 0, 0];
    for (let index = 0; index < changedIds.length; index++) {
        const id = changedIds[index];
        const bySphere = sphere.has(id) || frontVisible.has(id);
        const byPaintThrough = paintThrough.has(id);
        const group = bySphere ? (byPaintThrough ? 2 : 0) : (byPaintThrough ? 1 : 3);
        groups[group][offsets[group]++] = id;
    }

    return {
        changedCount: changedIds.length,
        frontVisibleSupplement,
        frontVisibleSupplementOnly,
        sphereOnly: groups[0],
        paintThroughOnly: groups[1],
        overlap: groups[2],
        unclassified: groups[3]
    };
};

const selectEvenlySpacedIds = (ids: ArrayLike<number>, maximum = 128) => {
    if (maximum <= 0 || ids.length === 0) return [];
    if (ids.length <= maximum) return Array.from(ids);
    if (maximum === 1) return [ids[0]];
    const result: number[] = [];
    for (let index = 0; index < maximum; index++) {
        result.push(ids[Math.round(index * (ids.length - 1) / (maximum - 1))]);
    }
    return result;
};

class PaintDiagnosticSampleCollector {
    readonly maximum: number;
    private stride = 1;
    private total = 0;
    private retained: PaintDiagnosticSample[] = [];
    private last: PaintDiagnosticSample | null = null;

    constructor(maximum = 500) {
        this.maximum = Math.max(2, Math.floor(maximum));
    }

    add(sample: PaintDiagnosticSample) {
        this.total++;
        this.last = sample;
        if (sample.sequence % this.stride === 0) this.retained.push(sample);
        while (this.retained.length > this.maximum - 1) {
            this.stride *= 2;
            this.retained = this.retained.filter(value => value.sequence % this.stride === 0);
        }
    }

    snapshot() {
        const samples = [...this.retained];
        if (this.last && !samples.some(sample => sample.sequence === this.last?.sequence)) samples.push(this.last);
        samples.sort((a, b) => a.sequence - b.sequence);
        return {
            total: this.total,
            retained: samples.slice(0, this.maximum),
            omitted: Math.max(0, this.total - samples.length),
            strategy: 'first-last-and-power-of-two-even-spacing'
        };
    }
}

const buildPaintDiagnostic = (input: PaintDiagnosticBuildInput) => {
    const signals: PaintDiagnosticSignal[] = [];

    if (input.status === 'error') {
        signals.push({
            code: 'STROKE_ERROR',
            severity: 'error',
            evidence: input.error || input.reason || 'The stroke ended with an unspecified error.',
            interpretation: 'The stroke did not complete normally. Inspect the status and performance sections around the failing stage.'
        });
    }
    if (input.status === 'cancelled') {
        signals.push({
            code: 'STROKE_CANCELLED',
            severity: 'info',
            evidence: input.reason || 'The stroke was cancelled.',
            interpretation: 'A selection change, tool switch, pointer cancellation, or deactivation stopped this stroke before commit.'
        });
    }
    if (input.reason === 'active-paint-layer-not-visible') {
        signals.push({
            code: 'ACTIVE_PAINT_LAYER_NOT_VISIBLE',
            severity: 'warning',
            evidence: 'The pointer-down was rejected because the active paint layer was not visible.',
            interpretation: 'Make the active layer visible before painting. No surface picking or GPU brush work was attempted.'
        });
    }
    if (input.sampleTotals.target === 0 && input.status !== 'cancelled' && input.status !== 'error') {
        signals.push({
            code: 'NO_EDITABLE_SURFACE_HIT',
            severity: 'warning',
            evidence: `None of the ${input.totalSamples} processed samples produced a target-surface hit.`,
            interpretation: 'The brush could not establish a paint center. Check locked, deleted, solo-hidden, independently edited, or low-opacity target data.'
        });
    }
    if (input.sampleTotals.otherSplat > 0) {
        signals.push({
            code: 'OTHER_SPLAT_BLOCKED_STROKE',
            severity: 'warning',
            evidence: `${input.sampleTotals.otherSplat} processed samples hit another visible splat before the selected paint target.`,
            interpretation: 'Those samples were intentionally rejected, which can look like an unpaintable area.'
        });
    }
    if (input.sourceClassification.changedCount === 0 && input.status !== 'cancelled' && input.status !== 'error') {
        signals.push({
            code: 'ZERO_CHANGED_GAUSSIANS',
            severity: 'warning',
            evidence: 'The committed stroke changed zero Gaussians.',
            interpretation: 'Surface picking may have succeeded while every candidate was filtered or the GPU brush produced an empty overlay.'
        });
    }
    if (input.sampleTotals.paintThroughBehindSurface > 0) {
        signals.push({
            code: 'PAINT_THROUGH_BEHIND_SURFACE',
            severity: 'warning',
            evidence: `${input.sampleTotals.paintThroughBehindSurface} processed paint-through samples referenced Gaussian centers behind the picked surface; maximum signed depth was ${round(input.sampleTotals.maximumPaintThroughDepthBehindSurface)} world units.`,
            interpretation: 'If those IDs appear in paintThroughOnly or overlap, the explicit paint-through path contributed to rear-surface painting.'
        });
    }
    const coalescedRatio = input.inputEvents > 0 ? input.coalescedInputEvents / input.inputEvents : 0;
    if (input.coalescedInputEvents >= 5 && coalescedRatio >= 0.2) {
        signals.push({
            code: 'INPUT_EVENTS_COALESCED',
            severity: 'info',
            evidence: `${input.coalescedInputEvents} of ${input.inputEvents} queued input events (${round(coalescedRatio * 100, 2)}%) were replaced by a newer point while GPU work was pending.`,
            interpretation: 'The queue deliberately keeps its newest point. A high ratio can reduce path detail when picking or readback is slow.'
        });
    }
    const filtered = input.context.target.gaussianStates;
    if (filtered.locked + filtered.deleted + filtered.soloHidden + filtered.independentlyEdited > 0) {
        signals.push({
            code: 'TARGET_CONTAINS_FILTERED_GAUSSIANS',
            severity: 'info',
            evidence: `Target totals: locked=${filtered.locked}, deleted=${filtered.deleted}, soloHidden=${filtered.soloHidden}, independentlyEdited=${filtered.independentlyEdited}.`,
            interpretation: 'These counts are contextual rather than proof for a specific sample. Locked/deleted/solo-hidden points cannot be sphere-painted; independently edited points are excluded from the ID-pick path.'
        });
    }

    const depthPick = summarizeNumbers(input.samples.map(sample => sample.timings.depthPickMs));
    const idReadback = summarizeNumbers(input.samples.map(sample => sample.timings.idReadbackMs));
    const sampleProcessing = summarizeNumbers(input.samples.map(sample => sample.timings.processingMs));
    const stages = [
        { name: 'depthPick', totalMs: depthPick?.totalMs ?? 0 },
        { name: 'idReadback', totalMs: idReadback?.totalMs ?? 0 },
        { name: 'gpuReadback', totalMs: input.performance.gpuReadbackMs },
        { name: 'commitCpu', totalMs: input.performance.commitCpuMs }
    ].sort((a, b) => b.totalMs - a.totalMs);

    const source = input.sourceClassification;
    const statusSummary = input.status === 'success' ? 'completed successfully' :
        (input.status === 'empty' ? 'completed without changing any Gaussians' : `ended as ${input.status}`);
    const chatgptSummary = [
        `This diagnostic describes one paintbrush stroke on "${input.context.splat.filename || input.context.splat.name}" and ${statusSummary}.`,
        `The tool received ${input.inputEvents} input points, processed ${input.totalSamples}, retained ${input.samples.length} representative samples, and changed ${source.changedCount} Gaussians.`,
        `Changed-source classification (legacy v1 keys; primary mode=${input.context.brush.selectionMode ?? 'modelSphere'}): sphereOnly=${source.sphereOnly.length}, paintThroughOnly=${source.paintThroughOnly.length}, overlap=${source.overlap.length}, unclassified=${source.unclassified.length}.`,
        ...(input.context.picking.frontVisibleFootprintSupplementEnabled ? [
            `The front-visible footprint supplement contributed ${source.frontVisibleSupplement} changed Gaussians, including ${source.frontVisibleSupplementOnly} that the center-based primary pass did not reach.`
        ] : []),
        `The largest measured stage was ${stages[0].name} at ${round(stages[0].totalMs)} ms in the retained/commit measurements.`,
        signals.length > 0 ? `Objective signals: ${signals.map(signal => signal.code).join(', ')}.` : 'No objective warning signal was detected by the diagnostic formatter.'
    ];

    return {
        schema: 'resplat-paint-brush-diagnostic',
        version: 1,
        purpose: 'ChatGPT-readable evidence for paintbrush misses, unintended rear-surface painting, and performance bottlenecks.',
        createdAt: input.createdAt,
        completedAt: input.completedAt,
        units: {
            screenCoordinates: 'normalized 0..1 from the top-left of the visible paint canvas',
            positions: 'scene world units or splat-local model units as named',
            cameraDepth: 'world units along the camera forward axis',
            duration: 'milliseconds'
        },
        chatgptSummary,
        signals,
        status: {
            value: input.status,
            reason: input.reason ?? null,
            error: input.error ?? null
        },
        context: input.context,
        metrics: {
            stroke: {
                inputEvents: input.inputEvents,
                processedSamples: input.totalSamples,
                retainedSamples: input.samples.length,
                omittedSamples: Math.max(0, input.totalSamples - input.samples.length),
                targetSamples: input.sampleTotals.target,
                missedSamples: input.sampleTotals.miss,
                otherSplatSamples: input.sampleTotals.otherSplat,
                paintThroughBehindSurfaceSamples: input.sampleTotals.paintThroughBehindSurface,
                coalescedInputEvents: input.coalescedInputEvents,
                coalescedInputRatio: round(coalescedRatio)
            },
            paintSources: {
                changed: source.changedCount,
                frontVisibleSupplement: source.frontVisibleSupplement,
                frontVisibleSupplementOnly: source.frontVisibleSupplementOnly,
                sphereOnly: source.sphereOnly.length,
                paintThroughOnly: source.paintThroughOnly.length,
                overlap: source.overlap.length,
                unclassified: source.unclassified.length
            },
            modelRadius: summarizeDistribution(input.samples.flatMap(sample => (
                sample.brush ? [sample.brush.modelRadius] : []
            ))),
            frontSurfaceDepthToleranceWorldUnits: summarizeDistribution(input.samples.flatMap(sample => (
                sample.brush?.frontSurfaceDepthToleranceWorldUnits === undefined ? [] :
                    [sample.brush.frontSurfaceDepthToleranceWorldUnits]
            ))),
            performance: {
                depthPick,
                idReadback,
                sampleProcessing,
                queueDrainMs: round(input.performance.queueDrainMs),
                commitMs: round(input.performance.commitMs),
                gpuReadbackMs: round(input.performance.gpuReadbackMs),
                commitCpuMs: round(input.performance.commitCpuMs),
                totalStrokeMs: round(input.performance.totalStrokeMs),
                largestMeasuredStage: stages[0]
            }
        },
        sampleRetention: {
            total: input.totalSamples,
            retained: input.samples.length,
            omitted: Math.max(0, input.totalSamples - input.samples.length),
            maximum: 500,
            strategy: 'first-last-and-power-of-two-even-spacing'
        },
        samples: input.samples,
        representativeGaussians: input.representativeGaussians,
        analysisGuide: [
            'Start with chatgptSummary and signals, then verify each signal against metrics and samples.',
            'brush.strength is the user-facing control value. brush.effectiveStrength, when present, is the actual center opacity after the brush coverage response is applied.',
            'Painting attenuates view-dependent spherical harmonics by the accumulated paint alpha so the reported changed color is not visually overridden by retained SH detail.',
            'A positive paintThrough.depthBehindSurface means that the referenced Gaussian center is behind the depth-picked surface along the camera forward direction.',
            input.context.brush.selectionMode === 'screenCircleFrontSurface' ?
                'For schema v1 compatibility, sphereOnly is the legacy key for the primary brush path. In this report it means the screen-space circle plus front-surface depth filter changed the Gaussian.' :
                'sphereOnly means the 3D model-space brush sphere changed the Gaussian without help from the explicit paint-through ID sample.',
            'frontSurfaceDepthToleranceWorldUnits is the allowed distance behind the composited front-depth map at each Gaussian screen position. Nearer Gaussian centers are not rejected.',
            'When picking.frontVisibleFootprintSupplementEnabled is true, the primary path also includes front-visible Gaussian footprints touched by the screen-space stroke, even when their centers fall outside the brush circle.',
            'metrics.paintSources.frontVisibleSupplementOnly counts changed Gaussians found only by that footprint supplement; it is direct evidence that visible contributors would otherwise have been missed by center-based selection.',
            'paintThroughOnly means only the low-alpha ID-pick path added the Gaussian. overlap means both paths reached it.',
            'Locked, deleted, and solo-hidden Gaussians are rejected by the primary brush path. The ID-pick path also rejects independently edited/desaturated Gaussians.',
            'Representative Gaussian arrays are deliberately capped; their category counts in metrics.paintSources describe the complete committed stroke.',
            'This report contains evidence, not an automatic root-cause verdict. Compare multiple signals before proposing a fix.'
        ],
        optionalAttachments: [
            'A screen recording from before pointer-down until after the stroke becomes visible.',
            'A screenshot from the same camera pose.',
            'The original PLY when exact Gaussian geometry must be reproduced.'
        ]
    };
};

const paintDiagnosticFilename = (splatFilename: string | undefined, createdAt: string) => {
    const base = (splatFilename || 'scene').replace(/\.[^.]+$/, '').replace(/[^\w\u4e00-\u9fff-]+/g, '-');
    const timestamp = createdAt.replace(/[:.]/g, '-');
    return `ReSplat-Paint-Diagnostic-${base || 'scene'}-${timestamp}.json`;
};

export {
    PaintDiagnosticSampleCollector,
    buildPaintDiagnostic,
    classifyPaintSources,
    paintDiagnosticFilename,
    selectEvenlySpacedIds,
    summarizeNumbers
};
export type {
    DiagnosticVec3,
    PaintDiagnosticBuildInput,
    PaintDiagnosticContext,
    PaintDiagnosticGaussian,
    PaintDiagnosticSample,
    PaintDiagnosticSource,
    PaintDiagnosticStatus,
    PaintSourceClassification,
    TimingSummary
};
