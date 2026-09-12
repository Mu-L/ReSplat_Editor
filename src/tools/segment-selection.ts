import { Button, Container, Element, Label } from '@playcanvas/pcui';
import { Mat4, Quat, Vec3 } from 'playcanvas';

import { Events } from '../events';
import { Scene } from '../scene';
import {
    applyRefineMask,
    accumulateSegmentObservation,
    chooseStablePromptSample,
    createSegmentEvidenceAccumulator,
    evaluateAuxiliaryViewSafety,
    evaluateSegmentMaskQuality,
    filterAnchorConnectedSelection,
    finalizeSegmentEvidence,
    findMaskInteriorPoint,
    hasMaskHits,
    sampleDistributedMaskPoints,
    sampleMaskPoints,
    segmentAuxiliaryMinimumAreaRatio,
    selectSafestAuxiliaryPose,
    suppressGroundPlaneLeak,
    type SegmentOperation,
    type VisibilityObservation
} from '../segment-selection/mask-utils';
import {
    Sam2Runtime,
    type Sam2Prompt,
    type Sam2RuntimeStatus,
    type Sam2SegmentOptions
} from '../segment-selection/sam2-runtime';
import type { ResolvedMaskSelection, ResolveMaskOptions, SelectionMaskOperation } from '../selection-mask';
import { Splat } from '../splat';
import { State } from '../splat-state';
import { localize } from '../ui/localization';

type SegmentDimension = '2d' | '3d';
type SegmentDepthMode = 'surface' | 'through';
type NormalizedPoint = { x: number; y: number };
type ViewMask = {
    mask: Uint8Array;
    confidence: Uint8Array;
    predictedIou: number;
    width: number;
    height: number;
    diagnosticIndex?: number;
};
type SegmentResult = { resolved: ResolvedMaskSelection; degraded: boolean };
type SegmentAttempt = SegmentResult | 'empty' | null;
type SegmentDiagnosticSector = 'current' | 'left' | 'right' | 'back';
type SegmentDiagnosticViewMeta = {
    sector: SegmentDiagnosticSector;
    kind: 'primary' | 'base' | 'reroute';
    yaw?: number;
    pitch?: number;
};
type SegmentDiagnosticCamera = {
    position: [number, number, number];
    focalPoint: [number, number, number];
    fov: number;
    ortho: boolean;
};
type SegmentDiagnosticView = SegmentDiagnosticViewMeta & {
    attempt: number;
    prompt: NormalizedPoint;
    prompts: NormalizedPoint[];
    promptMode: 'single' | 'multi';
    camera: SegmentDiagnosticCamera;
    sourceSize: { width: number; height: number };
    predictedIou: number;
    maskPixels: number;
    maskRatio: number;
    imageDataUrl: string;
    maskConfidenceDataUrl: string;
    quality?: unknown;
    projections?: Record<string, { hits: number; visible?: number }>;
};
type SegmentDiagnostic = {
    schema: 'resplat-segment-select-diagnostic';
    version: 1;
    createdAt: string;
    completedAt?: string;
    splat: { filename: string; splats: number } | null;
    request: {
        operation: SegmentOperation;
        dimension: SegmentDimension;
        depthMode: SegmentDepthMode;
        point: NormalizedPoint;
        prompt?: NormalizedPoint;
        promptStable?: boolean;
        promptGaussianId?: number;
    };
    runtime: { backend: string | null; status: Sam2RuntimeStatus };
    originalCamera: SegmentDiagnosticCamera;
    restoredCamera?: SegmentDiagnosticCamera;
    poses: Array<SegmentDiagnosticViewMeta & {
        camera: SegmentDiagnosticCamera;
        safety: unknown;
        clearance: number | null;
        prompt?: NormalizedPoint;
    }>;
    views: SegmentDiagnosticView[];
    fusion?: Record<string, number | boolean>;
    result?: Record<string, unknown>;
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
// This removes low-contribution fragments from Segment Select's front-most ID
// pass without discarding the softer parts of normal Gaussian surfaces.
const segmentVisibleAlphaThreshold = 0.05;
const segmentOrbitViews = [
    { sector: 'left', angle: -45 },
    { sector: 'right', angle: 45 },
    { sector: 'back', angle: 180 }
] as const;
const segmentAuxiliaryMaximumPrompts = 5;
const diagnosticPreviewMaximumSize = 640;

const estimateMedianGaussianScale = (splat: Splat) => {
    const data = splat.splatData;
    const scale0 = data.getProp('scale_0') as Float32Array | undefined;
    const scale1 = data.getProp('scale_1') as Float32Array | undefined;
    const scale2 = data.getProp('scale_2') as Float32Array | undefined;
    if (!scale0 || !scale1 || !scale2 || data.numSplats === 0) return undefined;
    const sampleCount = Math.min(4096, data.numSplats);
    const step = Math.max(1, Math.floor(data.numSplats / sampleCount));
    const samples: number[] = [];
    for (let index = 0; index < data.numSplats && samples.length < sampleCount; index += step) {
        const value = Math.max(scale0[index], scale1[index], scale2[index]);
        if (Number.isFinite(value)) samples.push(Math.exp(Math.max(-30, Math.min(30, value))));
    }
    if (samples.length === 0) return undefined;
    samples.sort((a, b) => a - b);
    return samples[Math.floor(samples.length / 2)];
};

const countMaskHits = (mask?: ArrayLike<number>) => {
    if (!mask) return 0;
    let count = 0;
    for (let index = 0; index < mask.length; index++) if (mask[index] !== 0) count++;
    return count;
};

const diagnosticCameraSnapshot = (scene: Scene): SegmentDiagnosticCamera => ({
    position: [scene.camera.position.x, scene.camera.position.y, scene.camera.position.z],
    focalPoint: [scene.camera.focalPoint.x, scene.camera.focalPoint.y, scene.camera.focalPoint.z],
    fov: scene.camera.fov,
    ortho: scene.camera.ortho
});

const encodeDiagnosticPreviews = (
    rgba: Uint8Array,
    mask: Uint8Array,
    confidence: Uint8Array,
    width: number,
    height: number,
    prompts: readonly NormalizedPoint[]
) => {
    const scale = Math.min(1, diagnosticPreviewMaximumSize / Math.max(width, height));
    const previewWidth = Math.max(1, Math.round(width * scale));
    const previewHeight = Math.max(1, Math.round(height * scale));
    const encode = (source: 'image' | 'mask') => {
        const canvas = document.createElement('canvas');
        canvas.width = previewWidth;
        canvas.height = previewHeight;
        const context = canvas.getContext('2d');
        if (!context) return '';
        const image = context.createImageData(previewWidth, previewHeight);
        for (let y = 0; y < previewHeight; y++) {
            const sourceY = Math.min(height - 1, Math.floor((y + 0.5) / scale));
            for (let x = 0; x < previewWidth; x++) {
                const sourceX = Math.min(width - 1, Math.floor((x + 0.5) / scale));
                const sourceIndex = sourceY * width + sourceX;
                const targetIndex = (y * previewWidth + x) * 4;
                if (source === 'image') {
                    const rgbaIndex = sourceIndex * 4;
                    image.data[targetIndex] = rgba[rgbaIndex];
                    image.data[targetIndex + 1] = rgba[rgbaIndex + 1];
                    image.data[targetIndex + 2] = rgba[rgbaIndex + 2];
                } else if (mask[sourceIndex] !== 0) {
                    image.data[targetIndex] = 255;
                    image.data[targetIndex + 1] = confidence[sourceIndex];
                    image.data[targetIndex + 2] = 0;
                } else {
                    image.data[targetIndex] = confidence[sourceIndex];
                    image.data[targetIndex + 1] = confidence[sourceIndex];
                    image.data[targetIndex + 2] = confidence[sourceIndex];
                }
                image.data[targetIndex + 3] = 255;
            }
        }
        context.putImageData(image, 0, 0);
        if (source === 'image') {
            for (let index = 0; index < prompts.length; index++) {
                const markerX = clamp01(prompts[index].x) * previewWidth;
                const markerY = clamp01(prompts[index].y) * previewHeight;
                context.strokeStyle = index === 0 ? '#ff3050' : '#30d8ff';
                context.lineWidth = 2;
                context.beginPath();
                context.moveTo(markerX - 8, markerY);
                context.lineTo(markerX + 8, markerY);
                context.moveTo(markerX, markerY - 8);
                context.lineTo(markerX, markerY + 8);
                context.stroke();
            }
        }
        return canvas.toDataURL(source === 'image' ? 'image/jpeg' : 'image/png', 0.82);
    };
    return {
        imageDataUrl: encode('image'),
        maskConfidenceDataUrl: encode('mask')
    };
};

const diagnosticFilename = (splatFilename: string | undefined, createdAt: string) => {
    const base = (splatFilename || 'scene').replace(/\.[^.]+$/, '').replace(/[^\w\u4e00-\u9fff-]+/g, '-');
    const timestamp = createdAt.replace(/[:.]/g, '-');
    return `ReSplat-Segment-Diagnostic-${base || 'scene'}-${timestamp}.json`;
};

class SegmentSelection {
    activate: () => void;
    deactivate: () => void;
    preload: () => void;

    constructor(events: Events, scene: Scene, parent: HTMLElement, canvasContainer: Container) {
        let active = false;
        let busy = false;
        let pointerId: number | null = null;
        let pointerStart = { x: 0, y: 0 };
        let pointerMoved = false;
        let requestId = 0;
        let operation: SegmentOperation = 'set';
        let dimension: SegmentDimension = '3d';
        let depthMode: SegmentDepthMode = 'surface';
        let runtimeStatus: Sam2RuntimeStatus = { stage: 'idle' };
        let degradedStatusTimer: number | null = null;
        let diagnosticRecording = false;
        let activeDiagnostic: SegmentDiagnostic | null = null;
        let lastDiagnostic: SegmentDiagnostic | null = null;

        const exportLastDiagnostic = () => {
            if (!lastDiagnostic) return;
            const json = JSON.stringify(lastDiagnostic, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = diagnosticFilename(lastDiagnostic.splat?.filename, lastDiagnostic.createdAt);
            link.click();
            window.setTimeout(() => URL.revokeObjectURL(url), 0);
        };

        const maskCanvas = document.createElement('canvas');
        const maskContext = maskCanvas.getContext('2d');
        if (!maskContext) throw new Error('Unable to create Segment Select mask canvas');
        maskContext.globalCompositeOperation = 'copy';

        const selectToolbar = new Container({
            class: ['select-toolbar', 'segment-select-toolbar'],
            hidden: true
        });
        selectToolbar.dom.addEventListener('pointerdown', event => event.stopPropagation());

        const operationButtons = new Map<SegmentOperation, Button>();
        const makeOperationButton = (value: SegmentOperation, localeKey: string, width = 60) => {
            const button = new Button({
                text: localize(localeKey),
                width,
                class: 'select-toolbar-button'
            });
            button.on('click', () => {
                operation = value;
                updateButtonState();
            });
            operationButtons.set(value, button);
            selectToolbar.append(button);
        };

        makeOperationButton('set', 'toolbar.select.set');
        makeOperationButton('add', 'toolbar.select.add');
        makeOperationButton('remove', 'toolbar.select.remove', 70);
        selectToolbar.append(new Element({ class: 'select-toolbar-separator' }));

        const twoDButton = new Button({
            text: localize('toolbar.segment.2d'),
            width: 52,
            class: 'select-toolbar-button'
        });
        const threeDButton = new Button({
            text: localize('toolbar.segment.3d'),
            width: 52,
            class: 'select-toolbar-button'
        });
        twoDButton.on('click', () => {
            dimension = '2d';
            updateButtonState();
        });
        threeDButton.on('click', () => {
            dimension = '3d';
            // 3D has fixed volumetric candidate + visibility-carving semantics;
            // keep the 2D depth switch from implying that it changes 3D.
            depthMode = 'surface';
            updateButtonState();
        });
        selectToolbar.append(twoDButton);
        selectToolbar.append(threeDButton);
        const depthModeSeparator = new Element({ class: 'select-toolbar-separator' });
        selectToolbar.append(depthModeSeparator);

        const surfaceButton = new Button({
            text: localize('toolbar.segment.surface'),
            width: 64,
            class: 'select-toolbar-button'
        });
        const throughButton = new Button({
            text: localize('toolbar.segment.through'),
            width: 64,
            class: 'select-toolbar-button'
        });
        surfaceButton.on('click', () => {
            depthMode = 'surface';
            updateButtonState();
        });
        throughButton.on('click', () => {
            depthMode = 'through';
            updateButtonState();
        });
        selectToolbar.append(surfaceButton);
        selectToolbar.append(throughButton);

        const diagnosticRecordButton = new Button({
            text: localize('toolbar.segment.diagnostic.record'),
            width: 82,
            class: 'select-toolbar-button',
            hidden: true
        });
        const diagnosticExportButton = new Button({
            text: localize('toolbar.segment.diagnostic.export'),
            width: 82,
            class: 'select-toolbar-button',
            hidden: true
        });
        diagnosticRecordButton.on('click', () => {
            diagnosticRecording = !diagnosticRecording;
            updateButtonState();
        });
        diagnosticExportButton.on('click', exportLastDiagnostic);
        selectToolbar.append(diagnosticRecordButton);
        selectToolbar.append(diagnosticExportButton);

        const modelLabel = new Label({
            class: ['select-toolbar-label', 'segment-model-label'],
            text: localize('toolbar.segment.fast'),
            hidden: true
        });
        const statusLabel = new Label({
            class: ['select-toolbar-label', 'segment-status-label'],
            text: localize('toolbar.segment.status.ready'),
            hidden: true
        });
        selectToolbar.append(modelLabel);
        selectToolbar.append(statusLabel);
        canvasContainer.append(selectToolbar);

        const formatStatus = (status: Sam2RuntimeStatus) => {
            if (status.stage === 'loading') return localize('toolbar.segment.status.loading');
            if (status.stage === 'segmenting') return localize('toolbar.segment.status.segmenting');
            if (status.stage === 'error') return localize('toolbar.segment.status.error');
            if (status.stage === 'ready' && status.backend === 'wasm') return localize('toolbar.segment.status.wasm');
            if (status.stage === 'ready' && status.backend === 'webgpu') return localize('toolbar.segment.status.webgpu');
            return localize('toolbar.segment.status.ready');
        };

        const clearDegradedStatus = () => {
            if (degradedStatusTimer !== null) window.clearTimeout(degradedStatusTimer);
            degradedStatusTimer = null;
        };

        const runtime = new Sam2Runtime((status) => {
            runtimeStatus = status;
            clearDegradedStatus();
            statusLabel.text = formatStatus(status);
            if (busy) events.fire('spinnerText', statusLabel.text);
        });

        this.preload = () => {
            const loading = runtime.load();
            if (loading) {
                loading.catch((error) => {
                    // Keep startup non-blocking. The runtime publishes the
                    // error state and a later Segment Select operation retries.
                    console.error('[SAM2] Startup preload failed', error);
                });
            }
        };

        const showDegradedStatus = () => {
            clearDegradedStatus();
            statusLabel.text = localize('toolbar.segment.status.degraded');
            degradedStatusTimer = window.setTimeout(() => {
                degradedStatusTimer = null;
                if (active && !busy) statusLabel.text = formatStatus(runtimeStatus);
            }, 4000);
        };

        const showEmptyStatus = () => {
            clearDegradedStatus();
            statusLabel.text = localize('toolbar.segment.status.empty');
            degradedStatusTimer = window.setTimeout(() => {
                degradedStatusTimer = null;
                if (active && !busy) statusLabel.text = formatStatus(runtimeStatus);
            }, 4000);
        };

        const updateButtonState = () => {
            for (const [value, button] of operationButtons) {
                button.dom.classList.toggle('active', operation === value);
                button.enabled = !busy;
            }
            twoDButton.dom.classList.toggle('active', dimension === '2d');
            threeDButton.dom.classList.toggle('active', dimension === '3d');
            surfaceButton.dom.classList.toggle('active', depthMode === 'surface');
            throughButton.dom.classList.toggle('active', depthMode === 'through');
            depthModeSeparator.hidden = dimension === '3d';
            surfaceButton.hidden = dimension === '3d';
            throughButton.hidden = dimension === '3d';
            diagnosticRecordButton.dom.classList.toggle('active', diagnosticRecording);
            twoDButton.enabled = !busy;
            threeDButton.enabled = !busy;
            surfaceButton.enabled = !busy && dimension === '2d';
            throughButton.enabled = !busy && dimension === '2d';
            diagnosticRecordButton.enabled = !busy;
            diagnosticExportButton.enabled = !busy && lastDiagnostic !== null;
        };
        updateButtonState();

        const drawMask = (mask: Uint8Array, width: number, height: number) => {
            maskCanvas.width = width;
            maskCanvas.height = height;
            const image = maskContext.createImageData(width, height);
            for (let index = 0; index < mask.length; index++) {
                if (mask[index] === 0) continue;
                const pixel = index * 4;
                image.data[pixel] = 255;
                image.data[pixel + 3] = 255;
            }
            maskContext.putImageData(image, 0, 0);
        };

        const isCurrent = (id: number) => active && id === requestId;

        const stabilizePromptPoint = async (point: NormalizedPoint, splat: Splat, id: number) => {
            const { width, height } = scene.targetSize;
            if (width <= 0 || height <= 0) {
                if (activeDiagnostic) {
                    activeDiagnostic.request.prompt = { ...point };
                    activeDiagnostic.request.promptStable = false;
                }
                return { point, stable: false };
            }
            const sampleWidth = Math.min(9, width);
            const sampleHeight = Math.min(9, height);
            const px = Math.max(0, Math.min(width - sampleWidth, Math.floor(point.x * width) - Math.floor(sampleWidth / 2)));
            const py = Math.max(0, Math.min(height - sampleHeight, Math.floor(point.y * height) - Math.floor(sampleHeight / 2)));
            const read = async (alphaThreshold: number) => {
                scene.camera.pickPrep(splat, 'set', alphaThreshold);
                const raw = await scene.camera.pickRect(px / width, py / height, sampleWidth / width, sampleHeight / height);
                const normalized = new Uint32Array(sampleWidth * sampleHeight);
                normalized.fill(0xffffffff);
                for (let y = 0; y < sampleHeight; y++) {
                    for (let x = 0; x < sampleWidth; x++) {
                        normalized[y * sampleWidth + x] = raw[(sampleHeight - 1 - y) * sampleWidth + x] ?? 0xffffffff;
                    }
                }
                return normalized;
            };
            const low = await read(0.05);
            if (!isCurrent(id)) return { point, stable: false };
            const high = await read(0.15);
            if (!isCurrent(id)) return { point, stable: false };
            const stable = chooseStablePromptSample(low, high, sampleWidth, sampleHeight, {
                centers: splat.entity.gsplat.instance.sorter.centers,
                medianScale: estimateMedianGaussianScale(splat)
            });
            if (!stable?.stable) {
                if (activeDiagnostic) {
                    activeDiagnostic.request.prompt = { ...point };
                    activeDiagnostic.request.promptStable = false;
                    activeDiagnostic.request.promptGaussianId = stable?.id;
                }
                return { point, stable: false };
            }
            const stabilizedPoint = {
                x: clamp01((px + stable.x + 0.5) / width),
                y: clamp01((py + stable.y + 0.5) / height)
            };
            if (activeDiagnostic) {
                activeDiagnostic.request.prompt = stabilizedPoint;
                activeDiagnostic.request.promptStable = true;
                activeDiagnostic.request.promptGaussianId = stable.id;
            }
            return {
                point: stabilizedPoint,
                stable: true
            };
        };

        const hideDistractions = () => {
            const snapshot = {
                gridVisible: events.invoke('grid.visible') as boolean,
                gizmoKeyHidden: events.invoke('gizmo.keyHidden') as boolean
            };
            events.fire('grid.setVisible', false);
            events.fire('gizmo.setKeyHidden', true);
            scene.forceRender = true;
            return snapshot;
        };

        const restoreDistractions = (snapshot: { gridVisible: boolean; gizmoKeyHidden: boolean }) => {
            events.fire('grid.setVisible', snapshot.gridVisible);
            events.fire('gizmo.setKeyHidden', snapshot.gizmoKeyHidden);
            scene.forceRender = true;
        };

        const inferViewMask = async (
            prompt: Sam2Prompt,
            id: number,
            retryEmpty = false,
            diagnosticMeta?: SegmentDiagnosticViewMeta,
            segmentOptions: Sam2SegmentOptions = {}
        ): Promise<ViewMask | null> => {
            const points: readonly NormalizedPoint[] = Array.isArray(prompt) ? prompt : [prompt as NormalizedPoint];
            if (points.length === 0) return null;
            const primaryPoint = points[0];
            const width = Math.max(1, parent.clientWidth);
            const height = Math.max(1, parent.clientHeight);
            // Do not capture while a previous selection is still uploading its
            // state texture. This is especially important for rapid repeated
            // Segment Select operations.
            await scene.commandQueue.enqueue(() => {});
            if (!isCurrent(id)) return null;
            const attempts = retryEmpty ? 2 : 1;
            let prediction: ViewMask = {
                mask: new Uint8Array(width * height),
                confidence: new Uint8Array(width * height),
                predictedIou: 0,
                width,
                height
            };
            for (let attempt = 0; attempt < attempts; attempt++) {
                const rgba = await events.invoke('render.offscreen', width, height) as Uint8Array;
                if (!isCurrent(id)) return null;
                prediction = await runtime.segment(rgba, width, height, points, segmentOptions);
                if (!isCurrent(id)) return null;
                if (activeDiagnostic && diagnosticMeta) {
                    const previews = encodeDiagnosticPreviews(
                        rgba,
                        prediction.mask,
                        prediction.confidence,
                        width,
                        height,
                        points
                    );
                    prediction.diagnosticIndex = activeDiagnostic.views.push({
                        ...diagnosticMeta,
                        attempt: attempt + 1,
                        prompt: { ...primaryPoint },
                        prompts: points.map(point => ({ ...point })),
                        promptMode: points.length > 1 ? 'multi' : 'single',
                        camera: diagnosticCameraSnapshot(scene),
                        sourceSize: { width, height },
                        predictedIou: prediction.predictedIou,
                        maskPixels: countMaskHits(prediction.mask),
                        maskRatio: countMaskHits(prediction.mask) / prediction.mask.length,
                        ...previews
                    }) - 1;
                }
                if (hasMaskHits(prediction.mask)) break;
                scene.forceRender = true;
            }
            return prediction;
        };

        const resolveViewMask = async (
            view: ViewMask,
            options: ResolveMaskOptions,
            id: number
        ): Promise<ResolvedMaskSelection | null> => {
            drawMask(view.mask, view.width, view.height);
            try {
                const resolved = await events.invoke('select.resolveMask', maskCanvas, maskContext, {
                    ...options,
                    sourceConfidence: view.confidence,
                    predictedIou: view.predictedIou
                }) as
                    ResolvedMaskSelection | null;
                if (activeDiagnostic && view.diagnosticIndex !== undefined && resolved) {
                    const record = activeDiagnostic.views[view.diagnosticIndex];
                    if (record) {
                        record.projections ??= {};
                        record.projections[options.projection ?? 'auto'] = {
                            hits: countMaskHits(resolved.hits),
                            visible: resolved.visible ? countMaskHits(resolved.visible) : undefined
                        };
                    }
                }
                return isCurrent(id) ? resolved : null;
            } finally {
                maskContext.clearRect(0, 0, view.width, view.height);
            }
        };

        const resolveVisible = async (view: ViewMask, id: number, collectVisible = false) => {
            let resolved = await resolveViewMask(view, {
                projection: 'visible',
                alphaThreshold: segmentVisibleAlphaThreshold,
                collectVisible
            }, id);
            // A very soft but legitimate target may have no fragment above the
            // normal threshold. Retry once without filtering instead of making
            // a valid SAM mask look like an ignored click.
            if (isCurrent(id) && (!resolved || !hasMaskHits(resolved.hits))) {
                resolved = await resolveViewMask(view, {
                    projection: 'visible',
                    alphaThreshold: 0,
                    collectVisible
                }, id);
            }
            return resolved;
        };

        const segmentTwoD = async (point: NormalizedPoint, id: number): Promise<SegmentAttempt> => {
            const selectedSplat = events.invoke('splatSelection') as Splat;
            const prompt = selectedSplat instanceof Splat ?
                await stabilizePromptPoint(point, selectedSplat, id) : { point, stable: false };
            const view = await inferViewMask(prompt.point, id, true, { sector: 'current', kind: 'primary' });
            if (!view) return null;
            if (!hasMaskHits(view.mask)) return 'empty';
            const resolved = depthMode === 'surface' ?
                await resolveVisible(view, id) :
                await resolveViewMask(view, { projection: 'centers' }, id);
            if (!resolved || !hasMaskHits(resolved.hits)) return isCurrent(id) ? 'empty' : null;
            return { resolved, degraded: false };
        };

        const findAnchor = async (
            view: ViewMask,
            point: NormalizedPoint,
            splat: Splat,
            id: number,
            reliablePoint: boolean
        ) => {
            const candidates: NormalizedPoint[] = reliablePoint ? [point] : [];
            const highConfidenceMask = new Uint8Array(view.mask.length);
            for (let index = 0; index < highConfidenceMask.length; index++) {
                if (view.mask[index] !== 0 && view.confidence[index] >= Math.round(0.7 * 255)) {
                    highConfidenceMask[index] = 255;
                }
            }
            const highConfidenceInterior = findMaskInteriorPoint(
                highConfidenceMask,
                view.width,
                view.height,
                point
            );
            if (highConfidenceInterior) candidates.push(highConfidenceInterior);
            const interior = findMaskInteriorPoint(view.mask, view.width, view.height, point);
            if (interior) {
                candidates.push(interior);
                candidates.push(...sampleMaskPoints(view.mask, view.width, view.height, interior, 12));
            }
            if (!reliablePoint) candidates.push(point);

            const seen = new Set<string>();
            for (const candidate of candidates) {
                if (!isCurrent(id)) return null;
                const key = `${Math.round(candidate.x * view.width)}:${Math.round(candidate.y * view.height)}`;
                if (seen.has(key)) continue;
                seen.add(key);
                const hit = await scene.camera.intersect(
                    candidate.x,
                    candidate.y,
                    segmentVisibleAlphaThreshold,
                    [splat]
                );
                if (hit?.splat === splat) return hit;
            }
            return null;
        };

        const segmentThreeViews = async (point: NormalizedPoint, id: number): Promise<SegmentAttempt> => {
            const selectedSplat = events.invoke('splatSelection') as Splat;
            if (!(selectedSplat instanceof Splat) || !selectedSplat.visible) {
                throw new Error(localize('popup.segment.no-target'));
            }

            const prompt = await stabilizePromptPoint(point, selectedSplat, id);
            if (!isCurrent(id)) return null;
            const currentView = await inferViewMask(prompt.point, id, true, { sector: 'current', kind: 'primary' });
            if (!currentView) return null;
            if (!hasMaskHits(currentView.mask)) return 'empty';

            const currentVisible = await resolveVisible(currentView, id, true);
            if (!isCurrent(id)) return null;
            const currentCenters = await resolveViewMask(currentView, { projection: 'centers' }, id);
            if (!isCurrent(id)) return null;

            const currentSurface = currentVisible?.splat === selectedSplat && hasMaskHits(currentVisible.hits) ?
                currentVisible.hits : null;
            const fallback = currentSurface ?
                currentSurface :
                currentCenters?.splat === selectedSplat ? currentCenters.hits : null;
            if (!fallback || !hasMaskHits(fallback)) return 'empty';
            if (!currentCenters || currentCenters.splat !== selectedSplat || !hasMaskHits(currentCenters.hits)) {
                return { resolved: { splat: selectedSplat, hits: new Uint8Array(fallback) }, degraded: true };
            }
            if (!currentVisible?.visible) {
                return { resolved: { splat: selectedSplat, hits: new Uint8Array(fallback) }, degraded: true };
            }
            const currentMaskPixels = countMaskHits(currentView.mask);
            const auxiliarySegmentOptions: Sam2SegmentOptions = {
                referenceMaskRatio: currentMaskPixels / currentView.mask.length,
                minimumRelativeArea: segmentAuxiliaryMinimumAreaRatio,
                maximumMaskRatio: 0.68
            };
            const distributedPromptSamples = sampleDistributedMaskPoints(
                currentView.mask,
                currentView.width,
                currentView.height,
                prompt.point,
                segmentAuxiliaryMaximumPrompts
            );
            const evidence = createSegmentEvidenceAccumulator(currentCenters.hits);
            accumulateSegmentObservation(evidence, {
                inside: currentVisible.hits,
                visible: currentVisible.visible,
                coverage: currentVisible.coverage,
                confidence: currentVisible.confidence,
                predictedIou: currentVisible.predictedIou
            });

            const anchor = await findAnchor(currentView, prompt.point, selectedSplat, id, prompt.stable);
            if (!isCurrent(id)) return null;
            if (!anchor) {
                return { resolved: { splat: selectedSplat, hits: new Uint8Array(fallback) }, degraded: true };
            }

            const originalPosition = scene.camera.position.clone();
            const originalTarget = scene.camera.focalPoint.clone();
            const originalFov = scene.camera.fov;
            const originalOrtho = scene.camera.ortho;
            let skippedViews = 0;
            const medianScale = estimateMedianGaussianScale(selectedSplat) ?? 0;
            let targetRadius = medianScale * 24;

            try {
                const offset = originalPosition.clone().sub(anchor.position);
                const worldUp = new Vec3(0, 1, 0);
                const { centers } = selectedSplat.entity.gsplat.instance.sorter;
                const radiusSamples: number[] = [];
                const worldCenter = new Vec3();
                const radiusMask = currentSurface ?? fallback;
                for (let index = 0; index < radiusMask.length && radiusSamples.length < 4096; index++) {
                    if (radiusMask[index] === 0) continue;
                    const centerOffset = index * 3;
                    worldCenter.set(centers[centerOffset], centers[centerOffset + 1], centers[centerOffset + 2]);
                    selectedSplat.worldTransform.transformPoint(worldCenter, worldCenter);
                    const distance = worldCenter.distance(anchor.position);
                    if (Number.isFinite(distance)) radiusSamples.push(distance);
                }
                radiusSamples.sort((a, b) => a - b);
                targetRadius = Math.max(
                    medianScale * 24,
                    radiusSamples.length > 0 ? radiusSamples[Math.floor((radiusSamples.length - 1) * 0.9)] : 0
                );
                const auxiliarySeedPositions: Vec3[] = [anchor.position.clone()];
                const maximumSeedDistance = Math.max(targetRadius * 1.5, medianScale * 24);
                const minimumSeedSeparation = Math.max(targetRadius * 0.05, medianScale * 8);
                for (const sample of distributedPromptSamples) {
                    if (!isCurrent(id)) return null;
                    const hit = await scene.camera.intersect(
                        sample.x,
                        sample.y,
                        segmentVisibleAlphaThreshold,
                        [selectedSplat]
                    );
                    if (
                        hit?.splat !== selectedSplat ||
                        hit.position.distance(anchor.position) > maximumSeedDistance ||
                        auxiliarySeedPositions.some(position => position.distance(hit.position) < minimumSeedSeparation)
                    ) {
                        continue;
                    }
                    auxiliarySeedPositions.push(hit.position.clone());
                    if (auxiliarySeedPositions.length >= segmentAuxiliaryMaximumPrompts) break;
                }
                type PoseSpec = { yaw: number; pitch: number };
                type InspectedPose = {
                    spec: PoseSpec;
                    safety: ReturnType<typeof evaluateAuxiliaryViewSafety>;
                    clearance: number;
                    prompt?: NormalizedPoint;
                };
                const poseKey = (spec: PoseSpec) => `${spec.yaw}:${spec.pitch}`;
                const setOrbitPose = (spec: PoseSpec) => {
                    const rotated = new Quat().setFromAxisAngle(worldUp, spec.yaw).transformVector(offset.clone());
                    if (spec.pitch !== 0) {
                        const pitchAxis = new Vec3().cross(worldUp, rotated).normalize();
                        new Quat().setFromAxisAngle(pitchAxis, spec.pitch).transformVector(rotated, rotated);
                    }
                    scene.camera.setPose(anchor.position.clone().add(rotated), anchor.position, 0);
                    scene.camera.onUpdate(0);
                    scene.forceRender = true;
                };
                const inspectPose = async (
                    spec: PoseSpec,
                    diagnosticMeta: SegmentDiagnosticViewMeta
                ): Promise<InspectedPose> => {
                    setOrbitPose(spec);
                    const projected = new Vec3();
                    scene.camera.worldToScreen(anchor.position, projected);
                    const anchorInViewport = projected.z >= 0 && projected.x >= 0 && projected.x <= 1 &&
                        projected.y >= 0 && projected.y <= 1;
                    const snapped = anchorInViewport ? await scene.camera.intersect(
                        projected.x,
                        projected.y,
                        segmentVisibleAlphaThreshold,
                        [selectedSplat]
                    ) : null;
                    const safety = evaluateAuxiliaryViewSafety({
                        anchorInViewport,
                        cameraAnchorDistance: scene.camera.position.distance(anchor.position),
                        firstHitDistance: snapped?.distance,
                        firstHitAnchorDistance: snapped?.position.distance(anchor.position),
                        targetRadius,
                        medianScale
                    });
                    if (!safety.accepted || !snapped) {
                        const inspected = { spec, safety, clearance: snapped?.distance ?? -Infinity };
                        if (activeDiagnostic) {
                            activeDiagnostic.poses.push({
                                ...diagnosticMeta,
                                camera: diagnosticCameraSnapshot(scene),
                                safety,
                                clearance: Number.isFinite(inspected.clearance) ? inspected.clearance : null
                            });
                        }
                        return inspected;
                    }
                    scene.camera.worldToScreen(snapped.position, projected);
                    const prompt = projected.z >= 0 && projected.x >= 0 && projected.x <= 1 &&
                        projected.y >= 0 && projected.y <= 1 ? { x: projected.x, y: projected.y } : undefined;
                    const inspected: InspectedPose = {
                        spec,
                        safety: prompt ? safety : { accepted: false, reason: 'out-of-view' },
                        clearance: snapped.distance,
                        prompt
                    };
                    if (activeDiagnostic) {
                        activeDiagnostic.poses.push({
                            ...diagnosticMeta,
                            camera: diagnosticCameraSnapshot(scene),
                            safety: inspected.safety,
                            clearance: inspected.clearance,
                            prompt
                        });
                    }
                    return inspected;
                };
                const promptIsInside = (view: ViewMask, point: NormalizedPoint) => {
                    const centerX = Math.max(0, Math.min(view.width - 1, Math.floor(point.x * view.width)));
                    const centerY = Math.max(0, Math.min(view.height - 1, Math.floor(point.y * view.height)));
                    for (let y = Math.max(0, centerY - 2); y <= Math.min(view.height - 1, centerY + 2); y++) {
                        for (let x = Math.max(0, centerX - 2); x <= Math.min(view.width - 1, centerX + 2); x++) {
                            if (view.mask[y * view.width + x] !== 0) return true;
                        }
                    }
                    return false;
                };
                const buildAuxiliaryPrompts = async (primary: NormalizedPoint) => {
                    const result: NormalizedPoint[] = [{ ...primary }];
                    const projected = new Vec3();
                    for (const seed of auxiliarySeedPositions) {
                        if (result.length >= segmentAuxiliaryMaximumPrompts || !isCurrent(id)) break;
                        scene.camera.worldToScreen(seed, projected);
                        if (
                            projected.z < 0 || projected.x < 0 || projected.x > 1 ||
                            projected.y < 0 || projected.y > 1
                        ) {
                            continue;
                        }
                        const snapped = await scene.camera.intersect(
                            projected.x,
                            projected.y,
                            segmentVisibleAlphaThreshold,
                            [selectedSplat]
                        );
                        if (
                            snapped?.splat !== selectedSplat ||
                            snapped.position.distance(anchor.position) > maximumSeedDistance
                        ) {
                            continue;
                        }
                        scene.camera.worldToScreen(snapped.position, projected);
                        if (
                            projected.z < 0 || projected.x < 0 || projected.x > 1 ||
                            projected.y < 0 || projected.y > 1
                        ) {
                            continue;
                        }
                        const candidate = { x: projected.x, y: projected.y };
                        if (result.some(point => Math.hypot(point.x - candidate.x, point.y - candidate.y) < 0.025)) {
                            continue;
                        }
                        result.push(candidate);
                    }
                    return result;
                };
                const captureObservation = async (
                    inspected: InspectedPose,
                    diagnosticMeta: SegmentDiagnosticViewMeta
                ) => {
                    if (!inspected.safety.accepted || !inspected.prompt || !isCurrent(id)) return null;
                    setOrbitPose(inspected.spec);
                    const inferAndEvaluate = async (viewPrompt: Sam2Prompt) => {
                        const view = await inferViewMask(
                            viewPrompt,
                            id,
                            false,
                            diagnosticMeta,
                            auxiliarySegmentOptions
                        );
                        if (!view) return null;
                        const promptPoints: readonly NormalizedPoint[] = Array.isArray(viewPrompt) ?
                            viewPrompt : [viewPrompt as NormalizedPoint];
                        const quality = evaluateSegmentMaskQuality({
                            predictedIou: view.predictedIou,
                            maskPixels: countMaskHits(view.mask),
                            totalPixels: view.mask.length,
                            promptInside: promptPoints.every(point => promptIsInside(view, point)),
                            referenceMaskPixels: currentMaskPixels,
                            minimumRelativeArea: segmentAuxiliaryMinimumAreaRatio
                        });
                        if (activeDiagnostic && view.diagnosticIndex !== undefined) {
                            const record = activeDiagnostic.views[view.diagnosticIndex];
                            if (record) record.quality = quality;
                        }
                        return { view, quality };
                    };
                    let attempt = await inferAndEvaluate(inspected.prompt);
                    if (!attempt) return null;
                    if (attempt.quality.accepted === false && attempt.quality.reason === 'undersized-mask') {
                        const strongerPrompts = await buildAuxiliaryPrompts(inspected.prompt);
                        if (!isCurrent(id)) return null;
                        if (strongerPrompts.length > 1) attempt = await inferAndEvaluate(strongerPrompts);
                    }
                    if (!attempt || !attempt.quality.accepted) return null;
                    const { view } = attempt;
                    const resolved = await resolveVisible(view, id, true);
                    if (
                        resolved?.splat !== selectedSplat || !resolved.visible ||
                        !hasMaskHits(resolved.hits)
                    ) {
                        return null;
                    }
                    const projectedCenters = await resolveViewMask(view, { projection: 'centers' }, id);
                    const silhouette = projectedCenters?.splat === selectedSplat && hasMaskHits(projectedCenters.hits) ?
                        projectedCenters.hits : undefined;
                    return {
                        inside: resolved.hits,
                        visible: resolved.visible,
                        silhouette,
                        coverage: resolved.coverage,
                        confidence: resolved.confidence,
                        predictedIou: resolved.predictedIou
                    } satisfies VisibilityObservation;
                };
                for (const { angle, sector } of segmentOrbitViews) {
                    if (!isCurrent(id)) return null;
                    try {
                        const baseMeta: SegmentDiagnosticViewMeta = { sector, kind: 'base', yaw: angle, pitch: 0 };
                        const base = await inspectPose({ yaw: angle, pitch: 0 }, baseMeta);
                        let observation = await captureObservation(base, baseMeta);
                        if (!observation) {
                            const alternatives: PoseSpec[] = [
                                { yaw: angle - 15, pitch: 0 },
                                { yaw: angle + 15, pitch: 0 },
                                { yaw: angle, pitch: -20 },
                                { yaw: angle, pitch: 20 }
                            ];
                            const inspectedAlternatives: InspectedPose[] = [];
                            for (const alternative of alternatives) {
                                inspectedAlternatives.push(await inspectPose(alternative, {
                                    sector,
                                    kind: 'reroute',
                                    yaw: alternative.yaw,
                                    pitch: alternative.pitch
                                }));
                            }
                            const reroute = selectSafestAuxiliaryPose(inspectedAlternatives.map(candidate => ({
                                value: candidate,
                                safety: candidate.safety,
                                clearance: candidate.clearance
                            })));
                            if (reroute && poseKey(reroute.spec) !== poseKey(base.spec)) {
                                observation = await captureObservation(reroute, {
                                    sector,
                                    kind: 'reroute',
                                    yaw: reroute.spec.yaw,
                                    pitch: reroute.spec.pitch
                                });
                            }
                        }
                        if (observation) accumulateSegmentObservation(evidence, observation);
                        else skippedViews++;
                    } catch (error) {
                        skippedViews++;
                        console.warn(`[Segment Select] ${angle} degree side view was skipped`, error);
                    }
                }
            } finally {
                scene.camera.fov = originalFov;
                scene.camera.ortho = originalOrtho;
                scene.camera.setPose(originalPosition, originalTarget, 0);
                scene.camera.onUpdate(0);
                scene.forceRender = true;
            }

            if (!isCurrent(id)) return null;
            const fused = finalizeSegmentEvidence(evidence);
            if (activeDiagnostic) {
                activeDiagnostic.fusion = {
                    centerCandidates: countMaskHits(currentCenters.hits),
                    fusedCandidates: countMaskHits(fused.candidates),
                    trustedSupport: countMaskHits(fused.support),
                    silhouetteCore: countMaskHits(fused.silhouetteCore),
                    visibleOutside: countMaskHits(fused.visibleOutside),
                    acceptedAuxiliaryViews: segmentOrbitViews.length - skippedViews,
                    skippedAuxiliaryViews: skippedViews
                };
            }
            if (!hasMaskHits(fused.support)) {
                return {
                    resolved: { splat: selectedSplat, hits: new Uint8Array(fallback) },
                    degraded: true
                };
            }
            const { centers } = selectedSplat.entity.gsplat.instance.sorter;
            const splatData = selectedSplat.splatData;
            const scale0 = splatData.getProp('scale_0') as Float32Array | undefined;
            const scale1 = splatData.getProp('scale_1') as Float32Array | undefined;
            const scale2 = splatData.getProp('scale_2') as Float32Array | undefined;
            const red = splatData.getProp('f_dc_0') as Float32Array | undefined;
            const green = splatData.getProp('f_dc_1') as Float32Array | undefined;
            const blue = splatData.getProp('f_dc_2') as Float32Array | undefined;
            const rotation0 = splatData.getProp('rot_0') as Float32Array | undefined;
            const rotation1 = splatData.getProp('rot_1') as Float32Array | undefined;
            const rotation2 = splatData.getProp('rot_2') as Float32Array | undefined;
            const rotation3 = splatData.getProp('rot_3') as Float32Array | undefined;
            const localAnchor = new Vec3();
            const inverseWorld = new Mat4().copy(selectedSplat.worldTransform).invert();
            inverseWorld.transformPoint(anchor.position, localAnchor);
            const localRadiusSamples: number[] = [];
            const localRadiusMask = currentSurface ?? fallback;
            for (let index = 0; index < localRadiusMask.length && localRadiusSamples.length < 4096; index++) {
                if (localRadiusMask[index] === 0) continue;
                const offset = index * 3;
                localRadiusSamples.push(Math.hypot(
                    centers[offset] - localAnchor.x,
                    centers[offset + 1] - localAnchor.y,
                    centers[offset + 2] - localAnchor.z
                ));
            }
            localRadiusSamples.sort((a, b) => a - b);
            const localTargetRadius = Math.max(
                medianScale * 24,
                localRadiusSamples.length > 0 ?
                    localRadiusSamples[Math.floor((localRadiusSamples.length - 1) * 0.9)] : 0
            );
            const connected = filterAnchorConnectedSelection(fused.candidates, fused.support, {
                centers,
                scales: scale0 && scale1 && scale2 ? [scale0, scale1, scale2] : undefined,
                colors: red && green && blue ? [red, green, blue] : undefined
            }, localAnchor, fused.silhouetteCore);
            const localUp = new Vec3();
            inverseWorld.transformVector(new Vec3(0, 1, 0), localUp);
            const suppressed = suppressGroundPlaneLeak(
                connected.hits,
                fused.support,
                fused.positiveViews,
                fused.maxConfidence,
                fused.visibleOutside,
                {
                    centers,
                    scales: scale0 && scale1 && scale2 ? [scale0, scale1, scale2] : undefined,
                    rotations: rotation0 && rotation1 && rotation2 && rotation3 ?
                        [rotation0, rotation1, rotation2, rotation3] : undefined,
                    anchor: localAnchor,
                    up: localUp,
                    targetRadius: localTargetRadius,
                    medianScale
                }
            );
            const connectedHits = hasMaskHits(suppressed.hits) ? suppressed.hits : fallback;
            if (activeDiagnostic?.fusion) {
                Object.assign(activeDiagnostic.fusion, {
                    connected: countMaskHits(connected.hits),
                    groundPlaneDetected: suppressed.detected,
                    groundPointsRemoved: suppressed.removed,
                    final3d: countMaskHits(connectedHits)
                });
            }
            return {
                resolved: { splat: selectedSplat, hits: connectedHits },
                degraded: skippedViews > 0 || connected.degraded || connectedHits === fallback
            };
        };

        const applyOperation = async (resolved: ResolvedMaskSelection, selectedOperation: SegmentOperation) => {
            let op: SelectionMaskOperation = selectedOperation === 'refine' ? 'set' : selectedOperation;
            let hits = resolved.hits;
            if (selectedOperation === 'refine') {
                const state = resolved.splat.splatData.getProp('state') as Uint8Array;
                hits = applyRefineMask(hits, state, State.selected);
                op = 'set';
            }
            return await events.invoke('select.applyResolvedMask', op, { splat: resolved.splat, hits });
        };

        const showError = async (error: unknown) => {
            await events.invoke('showPopup', {
                type: 'error',
                header: localize('popup.segment.error-title'),
                message: error instanceof Error ? error.message : String(error)
            });
        };

        const run = async (point: NormalizedPoint, selectedOperation: SegmentOperation) => {
            if (busy || !active) return;

            if (!events.invoke('selection.hasSplat')) {
                await showError(new Error(localize('popup.segment.no-target')));
                return;
            }

            const id = ++requestId;
            const selectedSplat = events.invoke('splatSelection') as Splat;
            activeDiagnostic = diagnosticRecording ? {
                schema: 'resplat-segment-select-diagnostic',
                version: 1,
                createdAt: new Date().toISOString(),
                splat: selectedSplat instanceof Splat ? {
                    filename: selectedSplat.filename,
                    splats: selectedSplat.splatData.numSplats
                } : null,
                request: {
                    operation: selectedOperation,
                    dimension,
                    depthMode,
                    point: { ...point }
                },
                runtime: {
                    backend: runtime.activeBackend,
                    status: { ...runtimeStatus }
                },
                originalCamera: diagnosticCameraSnapshot(scene),
                poses: [],
                views: [],
                result: { status: 'running' }
            } : null;
            let currentError: unknown = null;
            let degraded = false;
            let emptyResult = false;
            clearDegradedStatus();
            const distractionSnapshot = hideDistractions();
            busy = true;
            updateButtonState();
            events.fire('startSpinner');
            events.fire('spinnerText', localize('toolbar.segment.status.loading'));

            try {
                let result: SegmentAttempt;
                if (dimension === '3d' && !scene.camera.ortho) {
                    result = await segmentThreeViews(point, id);
                } else {
                    result = await segmentTwoD(point, id);
                    if (dimension === '3d') degraded = true;
                }
                if (!result || !isCurrent(id)) return;
                if (result === 'empty') {
                    emptyResult = true;
                    if (activeDiagnostic) {
                        activeDiagnostic.result = {
                            status: 'empty',
                            finalHits: 0,
                            degraded
                        };
                    }
                } else if (!hasMaskHits(result.resolved.hits)) {
                    emptyResult = true;
                    if (activeDiagnostic) {
                        activeDiagnostic.result = {
                            status: 'empty',
                            finalHits: 0,
                            degraded: degraded || result.degraded
                        };
                    }
                } else {
                    await applyOperation(result.resolved, selectedOperation);
                    degraded = degraded || result.degraded;
                    if (activeDiagnostic) {
                        activeDiagnostic.result = {
                            status: 'success',
                            finalHits: countMaskHits(result.resolved.hits),
                            degraded
                        };
                    }
                }
            } catch (error) {
                if (isCurrent(id)) currentError = error;
                if (activeDiagnostic) {
                    activeDiagnostic.result = isCurrent(id) ? {
                        status: 'error',
                        message: error instanceof Error ? error.message : String(error)
                    } : { status: 'cancelled' };
                }
            } finally {
                events.fire('stopSpinner');
                restoreDistractions(distractionSnapshot);
                if (activeDiagnostic) {
                    if (activeDiagnostic.result?.status === 'running') {
                        activeDiagnostic.result = { status: 'cancelled' };
                    }
                    activeDiagnostic.completedAt = new Date().toISOString();
                    activeDiagnostic.runtime = {
                        backend: runtime.activeBackend,
                        status: { ...runtimeStatus }
                    };
                    activeDiagnostic.restoredCamera = diagnosticCameraSnapshot(scene);
                    lastDiagnostic = activeDiagnostic;
                    activeDiagnostic = null;
                }
                busy = false;
                updateButtonState();
            }

            if (currentError !== null) {
                await showError(currentError);
            } else if (emptyResult) {
                showEmptyStatus();
            } else if (degraded) {
                showDegradedStatus();
            }
        };

        const pointerdown = (event: PointerEvent) => {
            if (busy) return;
            if (pointerId !== null || (event.pointerType === 'mouse' ? event.button !== 0 : !event.isPrimary)) return;
            pointerId = event.pointerId;
            pointerStart = { x: event.clientX, y: event.clientY };
            pointerMoved = false;
        };

        const pointermove = (event: PointerEvent) => {
            if (event.pointerId !== pointerId) return;
            if (Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5) pointerMoved = true;
        };

        const pointerup = (event: PointerEvent) => {
            if (event.pointerId !== pointerId) return;
            pointerId = null;
            if (pointerMoved) return;
            const rect = canvasContainer.dom.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            const point = {
                x: clamp01((event.clientX - rect.left) / rect.width),
                y: clamp01((event.clientY - rect.top) / rect.height)
            };
            const selectedOperation = event.shiftKey ? 'add' : (event.altKey ? 'remove' : operation);
            run(point, selectedOperation);
        };

        const pointercancel = (event: PointerEvent) => {
            if (event.pointerId === pointerId) pointerId = null;
        };

        this.activate = () => {
            active = true;
            parent.style.display = 'block';
            selectToolbar.hidden = false;
            canvasContainer.dom.addEventListener('pointerdown', pointerdown);
            canvasContainer.dom.addEventListener('pointermove', pointermove);
            canvasContainer.dom.addEventListener('pointerup', pointerup, true);
            canvasContainer.dom.addEventListener('pointercancel', pointercancel);
        };

        this.deactivate = () => {
            active = false;
            requestId++;
            pointerId = null;
            clearDegradedStatus();
            parent.style.display = 'none';
            selectToolbar.hidden = true;
            canvasContainer.dom.removeEventListener('pointerdown', pointerdown);
            canvasContainer.dom.removeEventListener('pointermove', pointermove);
            canvasContainer.dom.removeEventListener('pointerup', pointerup, true);
            canvasContainer.dom.removeEventListener('pointercancel', pointercancel);
        };

        window.addEventListener('beforeunload', () => {
            clearDegradedStatus();
            runtime.dispose();
        }, { once: true });
    }
}

export { SegmentSelection };
