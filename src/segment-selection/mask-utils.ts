import type { Sam2ImageTransform } from './sam2-image';

type SegmentOperation = 'set' | 'add' | 'remove' | 'refine';

type SelectedMask = {
    logits: Float32Array;
    width: number;
    height: number;
    candidate: number;
    predictedIou?: number;
};
type MaskCandidateSelectionOptions = {
    referenceMaskRatio?: number;
    minimumRelativeArea?: number;
    maximumMaskRatio?: number;
};
type SegmentPrediction = {
    mask: Uint8Array;
    confidence: Uint8Array;
    predictedIou: number;
    width: number;
    height: number;
};

const selectBestMask = (
    allLogits: Float32Array,
    dimensions: readonly number[],
    scores: Float32Array,
    options: MaskCandidateSelectionOptions = {}
): SelectedMask => {
    if (dimensions.length < 2) throw new Error('SAM2 mask output has invalid dimensions');
    const width = dimensions[dimensions.length - 1];
    const height = dimensions[dimensions.length - 2];
    const planeSize = width * height;
    if (planeSize <= 0 || allLogits.length % planeSize !== 0) {
        throw new Error('SAM2 mask output size does not match its dimensions');
    }

    const candidates = allLogits.length / planeSize;
    const referenceMaskRatio = Number.isFinite(options.referenceMaskRatio) ?
        Math.max(0, Math.min(1, options.referenceMaskRatio!)) : 0;
    const minimumRelativeArea = Number.isFinite(options.minimumRelativeArea) ?
        Math.max(0, options.minimumRelativeArea!) : 0;
    const maximumMaskRatio = Number.isFinite(options.maximumMaskRatio) ?
        Math.max(0, Math.min(1, options.maximumMaskRatio!)) : 1;
    let candidate = -1;
    let bestScore = -Infinity;
    for (let index = 0; index < Math.min(candidates, scores.length); index++) {
        let positive = 0;
        const offset = index * planeSize;
        for (let pixel = 0; pixel < planeSize; pixel++) {
            if (allLogits[offset + pixel] > 0) positive++;
        }
        const maskRatio = positive / planeSize;
        if (
            maskRatio > maximumMaskRatio ||
            (referenceMaskRatio > 0 && maskRatio < referenceMaskRatio * minimumRelativeArea)
        ) {
            continue;
        }
        if (scores[index] > bestScore) {
            bestScore = scores[index];
            candidate = index;
        }
    }
    // Preserve the decoder's normal highest-IoU behavior when every candidate
    // violates the expected-area guard. The caller will classify that mask as
    // undersized/oversized and retry it with stronger prompts without letting
    // it contribute negative visibility evidence.
    if (candidate === -1) {
        candidate = 0;
        bestScore = -Infinity;
        for (let index = 0; index < Math.min(candidates, scores.length); index++) {
            if (scores[index] > bestScore) {
                bestScore = scores[index];
                candidate = index;
            }
        }
    }

    return {
        logits: allLogits.slice(candidate * planeSize, (candidate + 1) * planeSize),
        width,
        height,
        candidate,
        predictedIou: Number.isFinite(scores[candidate]) ? scores[candidate] : 0
    };
};

const projectPredictionToSource = (mask: SelectedMask, transform: Sam2ImageTransform): SegmentPrediction => {
    const result: SegmentPrediction = {
        mask: new Uint8Array(transform.sourceWidth * transform.sourceHeight),
        confidence: new Uint8Array(transform.sourceWidth * transform.sourceHeight),
        predictedIou: mask.predictedIou ?? 0,
        width: transform.sourceWidth,
        height: transform.sourceHeight
    };
    for (let y = 0; y < transform.sourceHeight; y++) {
        const modelY = (y + 0.5) * transform.scaleY + transform.padY;
        const maskY = Math.max(0, Math.min(mask.height - 1, modelY / transform.modelSize * mask.height - 0.5));
        const y0 = Math.floor(maskY);
        const y1 = Math.min(mask.height - 1, y0 + 1);
        const fy = maskY - y0;
        for (let x = 0; x < transform.sourceWidth; x++) {
            const modelX = (x + 0.5) * transform.scaleX + transform.padX;
            const maskX = Math.max(0, Math.min(mask.width - 1, modelX / transform.modelSize * mask.width - 0.5));
            const x0 = Math.floor(maskX);
            const x1 = Math.min(mask.width - 1, x0 + 1);
            const fx = maskX - x0;
            const top = mask.logits[y0 * mask.width + x0] * (1 - fx) + mask.logits[y0 * mask.width + x1] * fx;
            const bottom = mask.logits[y1 * mask.width + x0] * (1 - fx) + mask.logits[y1 * mask.width + x1] * fx;
            const logit = top * (1 - fy) + bottom * fy;
            const index = y * transform.sourceWidth + x;
            if (logit > 0) result.mask[index] = 255;
            result.confidence[index] = Math.round(255 / (1 + Math.exp(-Math.max(-20, Math.min(20, logit)))));
        }
    }
    return result;
};

const projectMaskToSource = (mask: SelectedMask, transform: Sam2ImageTransform): Uint8Array => {
    return projectPredictionToSource(mask, transform).mask;
};

const hasMaskHits = (mask: Uint8Array) => {
    for (let index = 0; index < mask.length; index++) {
        if (mask[index] === 255) return true;
    }
    return false;
};

type NormalizedPoint = { x: number; y: number };
type VisibilityObservation = {
    inside: Uint8Array;
    visible: Uint8Array;
    silhouette?: Uint8Array;
    coverage?: Uint16Array;
    confidence?: Uint8Array;
    predictedIou?: number;
};
type SegmentFusion = {
    candidates: Uint8Array;
    support: Uint8Array;
    silhouetteCore: Uint8Array;
    positiveViews: Uint8Array;
    maxConfidence: Uint8Array;
    visibleOutside: Uint8Array;
};
type SegmentEvidenceAccumulator = {
    centerCandidates: Uint8Array;
    candidates: Uint8Array;
    positiveScore: Uint16Array;
    negativeScore: Uint16Array;
    positiveViews: Uint8Array;
    coreViews: Uint8Array;
    negativeViews: Uint8Array;
    maxConfidence: Uint8Array;
    visibleOutside: Uint8Array;
    silhouetteViews: Uint8Array;
    silhouetteObservations: number;
};
type CarvedSelection = { hits: Uint8Array; degraded: boolean };
type Point3 = { x: number; y: number; z: number };
type GaussianConnectivityData = {
    centers: ArrayLike<number>;
    scales?: readonly [ArrayLike<number>, ArrayLike<number>, ArrayLike<number>];
    colors?: readonly [ArrayLike<number>, ArrayLike<number>, ArrayLike<number>];
};
type ConnectivitySelection = { hits: Uint8Array; degraded: boolean };
type StablePromptData = {
    centers?: ArrayLike<number>;
    medianScale?: number;
};
type StablePromptSample = { id: number; x: number; y: number; stable: boolean };
type AuxiliaryViewRejectionReason = 'out-of-view' | 'no-surface' | 'camera-inside' | 'anchor-mismatch';
type AuxiliaryViewSafetyInput = {
    anchorInViewport: boolean;
    cameraAnchorDistance: number;
    firstHitDistance?: number;
    firstHitAnchorDistance?: number;
    targetRadius: number;
    medianScale: number;
};
type AuxiliaryViewSafety = { accepted: true } | { accepted: false; reason: AuxiliaryViewRejectionReason };
type AuxiliaryPoseCandidate<T> = { value: T; safety: AuxiliaryViewSafety; clearance: number };
type SegmentMaskRejectionReason =
    'empty-mask' | 'prompt-outside' | 'low-iou' | 'undersized-mask' | 'oversized-mask';
type SegmentMaskQualityInput = {
    predictedIou: number;
    maskPixels: number;
    totalPixels: number;
    promptInside: boolean;
    referenceMaskPixels?: number;
    minimumRelativeArea?: number;
};
type SegmentMaskQuality = { accepted: true } | { accepted: false; reason: SegmentMaskRejectionReason };
type GroundPlaneData = {
    centers: ArrayLike<number>;
    scales?: readonly [ArrayLike<number>, ArrayLike<number>, ArrayLike<number>];
    rotations?: readonly [ArrayLike<number>, ArrayLike<number>, ArrayLike<number>, ArrayLike<number>];
    anchor: Point3;
    up: Point3;
    targetRadius: number;
    medianScale: number;
};
type GroundPlaneSuppression = { hits: Uint8Array; detected: boolean; removed: number };

const SH_C0 = 0.28209479177387814;
const connectivityScaleSampleLimit = 4096;
const connectivityBaseDistanceFactor = 3;
const connectivityMaximumDistanceFactor = 6;
const connectivityMaximumScaleRatio = 8;
const connectivityColorDistance = 0.45;
const connectivityEnvelopeRatio = 0.25;
const connectivityEnvelopeMinimumScales = 12;
const connectivityMaximumGrowthHops = 3;
const connectivityMaximumGrowthRatio = 6;
const segmentAuxiliaryMinimumAreaRatio = 0.2;
const invalidPickId = 0xffffffff;

const chooseStablePromptSample = (
    lowAlphaIds: ArrayLike<number>,
    highAlphaIds: ArrayLike<number>,
    width: number,
    height: number,
    data: StablePromptData = {}
): StablePromptSample | null => {
    if (
        width <= 0 || height <= 0 ||
        lowAlphaIds.length !== width * height || highAlphaIds.length !== width * height
    ) {
        return null;
    }

    const centerX = Math.floor(width / 2);
    const centerY = Math.floor(height / 2);
    const byId = new Map<number, { low: number; high: number; x: number; y: number; distanceSq: number }>();
    const collect = (ids: ArrayLike<number>, high: boolean) => {
        for (let index = 0; index < ids.length; index++) {
            const id = ids[index];
            if (!Number.isInteger(id) || id < 0 || id === invalidPickId) continue;
            const x = index % width;
            const y = Math.floor(index / width);
            const distanceSq = (x - centerX) ** 2 + (y - centerY) ** 2;
            const current = byId.get(id) ?? { low: 0, high: 0, x, y, distanceSq };
            if (high) current.high++;
            else current.low++;
            if (distanceSq < current.distanceSq) {
                current.x = x;
                current.y = y;
                current.distanceSq = distanceSq;
            }
            byId.set(id, current);
        }
    };
    collect(lowAlphaIds, false);
    collect(highAlphaIds, true);
    if (byId.size === 0) return null;

    const centers = data.centers;
    const radius = Number.isFinite(data.medianScale) && data.medianScale! > 0 ? data.medianScale! * 6 : 0;
    const hasDenseNeighbors = (id: number) => {
        if (!centers || centers.length < (id + 1) * 3 || radius <= 0) return false;
        const offset = id * 3;
        const x = centers[offset];
        const y = centers[offset + 1];
        const z = centers[offset + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;
        const radiusSq = radius * radius;
        let neighbors = 0;
        // The 9x9 picker already gives a bounded local candidate set. Searching
        // the whole Splat here for every prompt candidate would turn one click
        // into O(81 × numSplats) work on large scenes.
        for (const other of byId.keys()) {
            if (other === id) continue;
            const otherOffset = other * 3;
            if (otherOffset + 2 >= centers.length) continue;
            const dx = centers[otherOffset] - x;
            const dy = centers[otherOffset + 1] - y;
            const dz = centers[otherOffset + 2] - z;
            if (dx * dx + dy * dy + dz * dz <= radiusSq && ++neighbors >= 6) return true;
        }
        return false;
    };

    const candidates = Array.from(byId, ([id, sample]) => ({
        id,
        ...sample,
        stable: sample.high >= 4 || hasDenseNeighbors(id)
    }));
    const centerId = lowAlphaIds[centerY * width + centerX];
    const center = candidates.find(candidate => candidate.id === centerId);
    if (center?.stable) return { id: center.id, x: center.x, y: center.y, stable: true };

    const stable = candidates
    .filter(candidate => candidate.stable && candidate.distanceSq <= 8 * 8)
    .sort((a, b) => a.distanceSq - b.distanceSq || b.high - a.high || b.low - a.low)[0];
    if (stable) return { id: stable.id, x: stable.x, y: stable.y, stable: true };

    const fallback = center ?? candidates.sort((a, b) => a.distanceSq - b.distanceSq || b.low - a.low)[0];
    return fallback ? { id: fallback.id, x: fallback.x, y: fallback.y, stable: false } : null;
};

const evaluateAuxiliaryViewSafety = (input: AuxiliaryViewSafetyInput): AuxiliaryViewSafety => {
    if (!input.anchorInViewport) return { accepted: false, reason: 'out-of-view' };
    if (
        !Number.isFinite(input.firstHitDistance) || input.firstHitDistance! < 0 ||
        !Number.isFinite(input.firstHitAnchorDistance) || input.firstHitAnchorDistance! < 0
    ) {
        return { accepted: false, reason: 'no-surface' };
    }
    const medianScale = Number.isFinite(input.medianScale) && input.medianScale > 0 ? input.medianScale : 0;
    const cameraAnchorDistance = Math.max(0, input.cameraAnchorDistance);
    const minimumClearance = Math.max(medianScale * 4, cameraAnchorDistance * 0.005);
    if (input.firstHitDistance! < minimumClearance) return { accepted: false, reason: 'camera-inside' };
    const targetRadius = Number.isFinite(input.targetRadius) && input.targetRadius > 0 ? input.targetRadius : 0;
    const maximumAnchorMismatch = Math.max(targetRadius * 1.5, medianScale * 24);
    if (input.firstHitAnchorDistance! > maximumAnchorMismatch) {
        return { accepted: false, reason: 'anchor-mismatch' };
    }
    return { accepted: true };
};

const selectSafestAuxiliaryPose = <T>(candidates: readonly AuxiliaryPoseCandidate<T>[]): T | null => {
    let best: AuxiliaryPoseCandidate<T> | null = null;
    for (const candidate of candidates) {
        if (!candidate.safety.accepted || !Number.isFinite(candidate.clearance)) continue;
        if (!best || candidate.clearance > best.clearance) best = candidate;
    }
    return best?.value ?? null;
};

const evaluateSegmentMaskQuality = (input: SegmentMaskQualityInput): SegmentMaskQuality => {
    if (input.totalPixels <= 0 || input.maskPixels <= 0) return { accepted: false, reason: 'empty-mask' };
    if (!input.promptInside) return { accepted: false, reason: 'prompt-outside' };
    if (!Number.isFinite(input.predictedIou) || input.predictedIou < 0.2) {
        return { accepted: false, reason: 'low-iou' };
    }
    const referenceMaskPixels = Number.isFinite(input.referenceMaskPixels) ?
        Math.max(0, input.referenceMaskPixels!) : 0;
    const minimumRelativeArea = Number.isFinite(input.minimumRelativeArea) ?
        Math.max(0, input.minimumRelativeArea!) :
        (referenceMaskPixels > 0 ? segmentAuxiliaryMinimumAreaRatio : 0);
    if (referenceMaskPixels > 0 && input.maskPixels < referenceMaskPixels * minimumRelativeArea) {
        return { accepted: false, reason: 'undersized-mask' };
    }
    if (input.maskPixels / input.totalPixels > 0.68) return { accepted: false, reason: 'oversized-mask' };
    return { accepted: true };
};

const findMaskInteriorPoint = (
    mask: Uint8Array,
    width: number,
    height: number,
    point: NormalizedPoint
): NormalizedPoint | null => {
    if (width <= 0 || height <= 0 || mask.length !== width * height || !hasMaskHits(mask)) return null;

    const startX = Math.max(0, Math.min(width - 1, Math.floor(point.x * width)));
    const startY = Math.max(0, Math.min(height - 1, Math.floor(point.y * height)));
    let start = startY * width + startX;

    if (mask[start] === 0) {
        let bestDistance = Infinity;
        for (let index = 0; index < mask.length; index++) {
            if (mask[index] === 0) continue;
            const x = index % width;
            const y = Math.floor(index / width);
            const distance = (x - startX) ** 2 + (y - startY) ** 2;
            if (distance < bestDistance) {
                bestDistance = distance;
                start = index;
            }
        }
    }

    const component = new Uint8Array(mask.length);
    const queue = new Int32Array(mask.length);
    let read = 0;
    let write = 0;
    queue[write++] = start;
    component[start] = 1;
    const visitComponent = (next: number) => {
        if (component[next] === 0 && mask[next] !== 0) {
            component[next] = 1;
            queue[write++] = next;
        }
    };

    while (read < write) {
        const index = queue[read++];
        const x = index % width;
        const y = Math.floor(index / width);
        if (x > 0) visitComponent(index - 1);
        if (x + 1 < width) visitComponent(index + 1);
        if (y > 0) visitComponent(index - width);
        if (y + 1 < height) visitComponent(index + width);
    }

    const distance = new Int32Array(mask.length);
    distance.fill(-1);
    read = 0;
    write = 0;
    for (let index = 0; index < component.length; index++) {
        if (component[index] === 0) continue;
        const x = index % width;
        const y = Math.floor(index / width);
        if (
            x === 0 || x + 1 === width || y === 0 || y + 1 === height ||
            component[index - 1] === 0 || component[index + 1] === 0 ||
            component[index - width] === 0 || component[index + width] === 0
        ) {
            distance[index] = 0;
            queue[write++] = index;
        }
    }

    let deepest = start;
    const visitDistance = (next: number, value: number) => {
        if (component[next] !== 0 && distance[next] === -1) {
            distance[next] = value;
            queue[write++] = next;
        }
    };
    while (read < write) {
        const index = queue[read++];
        if (distance[index] > distance[deepest]) deepest = index;
        const x = index % width;
        const y = Math.floor(index / width);
        const nextDistance = distance[index] + 1;
        if (x > 0) visitDistance(index - 1, nextDistance);
        if (x + 1 < width) visitDistance(index + 1, nextDistance);
        if (y > 0) visitDistance(index - width, nextDistance);
        if (y + 1 < height) visitDistance(index + width, nextDistance);
    }

    return {
        x: (deepest % width + 0.5) / width,
        y: (Math.floor(deepest / width) + 0.5) / height
    };
};

const sampleMaskPoints = (
    mask: Uint8Array,
    width: number,
    height: number,
    origin: NormalizedPoint,
    maximum = 16
) => {
    const result: NormalizedPoint[] = [];
    if (width <= 0 || height <= 0 || mask.length !== width * height || maximum <= 0) return result;
    const centerX = Math.max(0, Math.min(width - 1, Math.floor(origin.x * width)));
    const centerY = Math.max(0, Math.min(height - 1, Math.floor(origin.y * height)));

    const add = (x: number, y: number) => {
        if (x < 0 || x >= width || y < 0 || y >= height || mask[y * width + x] === 0) return;
        result.push({ x: (x + 0.5) / width, y: (y + 0.5) / height });
    };

    for (let radius = 0; radius < Math.max(width, height) && result.length < maximum; radius++) {
        if (radius === 0) {
            add(centerX, centerY);
            continue;
        }
        for (let x = centerX - radius; x <= centerX + radius && result.length < maximum; x++) {
            add(x, centerY - radius);
            if (result.length < maximum) add(x, centerY + radius);
        }
        for (let y = centerY - radius + 1; y < centerY + radius && result.length < maximum; y++) {
            add(centerX - radius, y);
            if (result.length < maximum) add(centerX + radius, y);
        }
    }
    return result;
};

const sampleDistributedMaskPoints = (
    mask: Uint8Array,
    width: number,
    height: number,
    origin: NormalizedPoint,
    maximum = 5
) => {
    if (width <= 0 || height <= 0 || mask.length !== width * height || maximum <= 0) return [];
    const deepest = findMaskInteriorPoint(mask, width, height, origin);
    if (!deepest) return [];

    // Keep prompt points on the clicked component. Detached mask islands are
    // frequently floaters or leaked ground and must not become positive SAM2
    // prompts in an auxiliary view.
    const component = new Uint8Array(mask.length);
    const queue = new Int32Array(mask.length);
    const startX = Math.max(0, Math.min(width - 1, Math.floor(deepest.x * width)));
    const startY = Math.max(0, Math.min(height - 1, Math.floor(deepest.y * height)));
    let read = 0;
    let write = 0;
    const start = startY * width + startX;
    component[start] = 255;
    queue[write++] = start;
    const visit = (index: number) => {
        if (mask[index] !== 0 && component[index] === 0) {
            component[index] = 255;
            queue[write++] = index;
        }
    };
    while (read < write) {
        const index = queue[read++];
        const x = index % width;
        const y = Math.floor(index / width);
        if (x > 0) visit(index - 1);
        if (x + 1 < width) visit(index + 1);
        if (y > 0) visit(index - width);
        if (y + 1 < height) visit(index + width);
    }

    const margin = Math.max(1, Math.floor(Math.min(width, height) * 0.01));
    const stride = Math.max(1, Math.floor(Math.sqrt(mask.length / 4096)));
    type Candidate = { x: number; y: number; point: NormalizedPoint };
    const candidates: Candidate[] = [];
    const collect = (requireMargin: boolean) => {
        candidates.length = 0;
        for (let y = 0; y < height; y += stride) {
            for (let x = 0; x < width; x += stride) {
                if (component[y * width + x] === 0) continue;
                if (requireMargin && (
                    x < margin || x + margin >= width || y < margin || y + margin >= height ||
                    component[y * width + x - margin] === 0 || component[y * width + x + margin] === 0 ||
                    component[(y - margin) * width + x] === 0 || component[(y + margin) * width + x] === 0
                )) {
                    continue;
                }
                candidates.push({ x, y, point: { x: (x + 0.5) / width, y: (y + 0.5) / height } });
            }
        }
    };
    collect(true);
    if (candidates.length < maximum) collect(false);

    const selected: NormalizedPoint[] = [deepest];
    const minimumDistanceSquared = 0.06 ** 2;
    while (selected.length < maximum && candidates.length > 0) {
        let best: Candidate | null = null;
        let bestDistance = -Infinity;
        for (const candidate of candidates) {
            let nearest = Infinity;
            for (const point of selected) {
                const dx = candidate.point.x - point.x;
                const dy = candidate.point.y - point.y;
                nearest = Math.min(nearest, dx * dx + dy * dy);
            }
            if (nearest > bestDistance) {
                bestDistance = nearest;
                best = candidate;
            }
        }
        if (!best || bestDistance < minimumDistanceSquared) break;
        selected.push(best.point);
        const selectedX = best.x;
        const selectedY = best.y;
        for (let index = candidates.length - 1; index >= 0; index--) {
            if (candidates[index].x === selectedX && candidates[index].y === selectedY) candidates.splice(index, 1);
        }
    }
    return selected;
};

const carveVisibleOutside = (candidates: Uint8Array, observations: readonly VisibilityObservation[]) => {
    const result = new Uint8Array(candidates);
    for (const observation of observations) {
        if (observation.inside.length !== result.length || observation.visible.length !== result.length) {
            throw new Error('Cannot carve visibility masks with different sizes');
        }
        for (let index = 0; index < result.length; index++) {
            if (result[index] !== 0 && observation.visible[index] !== 0 && observation.inside[index] === 0) {
                result[index] = 0;
            }
        }
    }
    return result;
};

const resolveCarvedSelection = (
    candidates: Uint8Array,
    currentVisible: Uint8Array,
    observations: readonly VisibilityObservation[],
    expectedObservations = 2
): CarvedSelection => {
    if (candidates.length !== currentVisible.length) {
        throw new Error('Current visible selection and center candidates have different sizes');
    }
    if (observations.length === 0) return { hits: new Uint8Array(currentVisible), degraded: true };
    const carved = carveVisibleOutside(candidates, observations);
    if (!hasMaskHits(carved)) return { hits: new Uint8Array(currentVisible), degraded: true };
    return { hits: carved, degraded: observations.length < expectedObservations };
};

const collectVisibilitySupport = (
    currentInside: Uint8Array,
    observations: readonly VisibilityObservation[]
) => {
    const support = new Uint8Array(currentInside);
    for (const observation of observations) {
        if (observation.inside.length !== support.length || observation.visible.length !== support.length) {
            throw new Error('Cannot collect visibility support from masks with different sizes');
        }
        for (let index = 0; index < support.length; index++) {
            if (observation.inside[index] !== 0) support[index] = 255;
        }
    }
    return support;
};

const createSegmentEvidenceAccumulator = (centerCandidates: Uint8Array): SegmentEvidenceAccumulator => {
    const count = centerCandidates.length;
    return {
        centerCandidates: new Uint8Array(centerCandidates),
        candidates: new Uint8Array(centerCandidates),
        positiveScore: new Uint16Array(count),
        negativeScore: new Uint16Array(count),
        positiveViews: new Uint8Array(count),
        coreViews: new Uint8Array(count),
        negativeViews: new Uint8Array(count),
        maxConfidence: new Uint8Array(count),
        visibleOutside: new Uint8Array(count),
        silhouetteViews: new Uint8Array(count),
        silhouetteObservations: 0
    };
};

const accumulateSegmentObservation = (
    accumulator: SegmentEvidenceAccumulator,
    observation: VisibilityObservation
) => {
    const count = accumulator.candidates.length;
    if (
        observation.inside.length !== count || observation.visible.length !== count ||
        (observation.silhouette && observation.silhouette.length !== count) ||
        (observation.confidence && observation.confidence.length !== count) ||
        (observation.coverage && observation.coverage.length !== count)
    ) {
        throw new Error('Cannot fuse segment observations with different sizes');
    }
    if (observation.silhouette) {
        accumulator.silhouetteObservations++;
        for (let index = 0; index < count; index++) {
            if (accumulator.centerCandidates[index] !== 0 && observation.silhouette[index] !== 0) {
                accumulator.silhouetteViews[index]++;
            }
        }
    }
    const quality = Number.isFinite(observation.predictedIou) ?
        Math.max(0.25, Math.min(1, observation.predictedIou!)) : 1;
    for (let index = 0; index < count; index++) {
        if (observation.visible[index] === 0) continue;
        const confidenceByte = observation.confidence?.[index] ?? (observation.inside[index] ? 255 : 0);
        accumulator.maxConfidence[index] = Math.max(accumulator.maxConfidence[index], confidenceByte);
        const confidence = confidenceByte / 255;
        if (observation.inside[index] !== 0) {
            accumulator.candidates[index] = 255;
            if (confidence >= 0.5) {
                accumulator.positiveViews[index]++;
                accumulator.positiveScore[index] += Math.round(confidence * quality * 255);
            }
            if (confidence >= 0.7) accumulator.coreViews[index]++;
            if (confidence <= 0.35) {
                accumulator.negativeViews[index]++;
                accumulator.negativeScore[index] += Math.round((1 - confidence) * quality * 255);
            }
        } else {
            accumulator.visibleOutside[index] = 255;
            accumulator.negativeViews[index]++;
            accumulator.negativeScore[index] += Math.round(Math.max(0.65, 1 - confidence) * quality * 255);
        }
    }
};

const finalizeSegmentEvidence = (accumulator: SegmentEvidenceAccumulator): SegmentFusion => {
    const {
        candidates,
        positiveScore,
        negativeScore,
        positiveViews,
        coreViews,
        negativeViews,
        maxConfidence,
        visibleOutside,
        silhouetteViews,
        silhouetteObservations
    } = accumulator;
    const support = new Uint8Array(candidates.length);
    const silhouetteCore = new Uint8Array(candidates.length);
    for (let index = 0; index < candidates.length; index++) {
        const strongMultiView = positiveViews[index] >= 2 && positiveScore[index] > negativeScore[index];
        const unopposedCore = coreViews[index] > 0 && negativeViews[index] === 0 && positiveScore[index] > negativeScore[index];
        if (strongMultiView || unopposedCore) support[index] = 255;
        if (negativeViews[index] > 0 && negativeScore[index] >= positiveScore[index] && !strongMultiView) {
            candidates[index] = 0;
            support[index] = 0;
        }
        if (silhouetteObservations >= 2 && silhouetteViews[index] >= 2 && candidates[index] !== 0) {
            silhouetteCore[index] = 255;
        }
    }
    return { candidates, support, silhouetteCore, positiveViews, maxConfidence, visibleOutside };
};

const fuseSegmentObservations = (
    centerCandidates: Uint8Array,
    observations: readonly VisibilityObservation[]
): SegmentFusion => {
    const accumulator = createSegmentEvidenceAccumulator(centerCandidates);
    for (const observation of observations) accumulateSegmentObservation(accumulator, observation);
    return finalizeSegmentEvidence(accumulator);
};

const suppressGroundPlaneLeak = (
    sourceHits: Uint8Array,
    support: Uint8Array,
    positiveViews: Uint8Array,
    maxConfidence: Uint8Array,
    visibleOutside: Uint8Array,
    data: GroundPlaneData
): GroundPlaneSuppression => {
    const count = sourceHits.length;
    if (
        support.length !== count || positiveViews.length !== count || maxConfidence.length !== count ||
        visibleOutside.length !== count ||
        data.centers.length < count * 3
    ) {
        throw new Error('Cannot suppress a ground plane with different sizes');
    }
    const upLength = Math.hypot(data.up.x, data.up.y, data.up.z);
    if (upLength === 0 || !Number.isFinite(upLength)) return { hits: new Uint8Array(sourceHits), detected: false, removed: 0 };
    const ux = data.up.x / upLength;
    const uy = data.up.y / upLength;
    const uz = data.up.z / upLength;
    const targetRadius = Math.max(0, data.targetRadius);
    const medianScale = Math.max(0, data.medianScale);
    const residualLimit = Math.max(medianScale * 2, targetRadius * 2 * 0.005);
    if (residualLimit <= 0 || targetRadius <= 0) {
        return { hits: new Uint8Array(sourceHits), detected: false, removed: 0 };
    }

    const hasNormals = Boolean(
        data.scales?.every(channel => channel.length >= count) &&
        data.rotations?.every(channel => channel.length >= count)
    );
    const normalMatchesUp = (index: number) => {
        if (!hasNormals || !data.scales || !data.rotations) return true;
        let axis = 0;
        if (data.scales[1][index] < data.scales[axis][index]) axis = 1;
        if (data.scales[2][index] < data.scales[axis][index]) axis = 2;
        let w = data.rotations[0][index];
        let x = data.rotations[1][index];
        let y = data.rotations[2][index];
        let z = data.rotations[3][index];
        const length = Math.hypot(w, x, y, z);
        if (!Number.isFinite(length) || length === 0) return false;
        w /= length;
        x /= length;
        y /= length;
        z /= length;
        let nx: number;
        let ny: number;
        let nz: number;
        if (axis === 0) {
            nx = 1 - 2 * (y * y + z * z);
            ny = 2 * (x * y + w * z);
            nz = 2 * (x * z - w * y);
        } else if (axis === 1) {
            nx = 2 * (x * y - w * z);
            ny = 1 - 2 * (x * x + z * z);
            nz = 2 * (y * z + w * x);
        } else {
            nx = 2 * (x * z + w * y);
            ny = 2 * (y * z - w * x);
            nz = 1 - 2 * (x * x + y * y);
        }
        return Math.abs(nx * ux + ny * uy + nz * uz) >= Math.cos(20 * Math.PI / 180);
    };

    const heights: number[] = [];
    const maximumLateral = targetRadius * 2.5;
    for (let index = 0; index < count && heights.length < 4096; index++) {
        if (visibleOutside[index] === 0 || !normalMatchesUp(index)) continue;
        const offset = index * 3;
        const dx = data.centers[offset] - data.anchor.x;
        const dy = data.centers[offset + 1] - data.anchor.y;
        const dz = data.centers[offset + 2] - data.anchor.z;
        const height = dx * ux + dy * uy + dz * uz;
        const lateralSq = dx * dx + dy * dy + dz * dz - height * height;
        if (lateralSq <= maximumLateral * maximumLateral && Math.abs(height) <= targetRadius * 2) heights.push(height);
    }
    if (heights.length < 64) return { hits: new Uint8Array(sourceHits), detected: false, removed: 0 };
    heights.sort((a, b) => a - b);
    const planeHeight = heights[Math.floor(heights.length / 2)];
    let inliers = 0;
    let squaredResidual = 0;
    for (const height of heights) {
        const residual = height - planeHeight;
        if (Math.abs(residual) <= residualLimit) {
            inliers++;
            squaredResidual += residual * residual;
        }
    }
    if (inliers < 64 || Math.sqrt(squaredResidual / inliers) > residualLimit) {
        return { hits: new Uint8Array(sourceHits), detected: false, removed: 0 };
    }

    const footprintSamples: number[] = [];
    const collectFootprint = (strongOnly: boolean) => {
        for (let index = 0; index < count; index++) {
            if (
                support[index] === 0 ||
                (strongOnly && positiveViews[index] <= 1 && maxConfidence[index] < Math.round(0.8 * 255))
            ) {
                continue;
            }
            const offset = index * 3;
            const dx = data.centers[offset] - data.anchor.x;
            const dy = data.centers[offset + 1] - data.anchor.y;
            const dz = data.centers[offset + 2] - data.anchor.z;
            const height = dx * ux + dy * uy + dz * uz;
            footprintSamples.push(Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - height * height)));
        }
    };
    collectFootprint(true);
    if (footprintSamples.length === 0) collectFootprint(false);
    footprintSamples.sort((a, b) => a - b);
    const footprintRadius = Math.max(
        medianScale * 6,
        footprintSamples.length > 0 ? footprintSamples[Math.floor((footprintSamples.length - 1) * 0.9)] : 0
    );
    const hits = new Uint8Array(sourceHits);
    let removed = 0;
    for (let index = 0; index < count; index++) {
        if (hits[index] === 0 || positiveViews[index] > 1 || maxConfidence[index] >= Math.round(0.8 * 255)) continue;
        const offset = index * 3;
        const dx = data.centers[offset] - data.anchor.x;
        const dy = data.centers[offset + 1] - data.anchor.y;
        const dz = data.centers[offset + 2] - data.anchor.z;
        const height = dx * ux + dy * uy + dz * uz;
        const lateral = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - height * height));
        if (Math.abs(height - planeHeight) <= residualLimit && lateral > footprintRadius + medianScale * 2) {
            hits[index] = 0;
            removed++;
        }
    }
    return { hits, detected: true, removed };
};

const filterAnchorConnectedSelection = (
    candidates: Uint8Array,
    support: Uint8Array,
    data: GaussianConnectivityData,
    anchor: Point3,
    silhouetteCore?: Uint8Array
): ConnectivitySelection => {
    const { centers, scales, colors } = data;
    const count = candidates.length;
    if (support.length !== count || (silhouetteCore && silhouetteCore.length !== count) || centers.length < count * 3) {
        throw new Error('Cannot filter connected Gaussians with different sizes');
    }
    if (scales?.some(channel => channel.length < count) || colors?.some(channel => channel.length < count)) {
        throw new Error('Cannot filter connected Gaussian properties with different sizes');
    }

    let candidateCount = 0;
    let supportedCount = 0;
    let seed = -1;
    let seedDistanceSq = Infinity;
    let supportMinX = Infinity;
    let supportMinY = Infinity;
    let supportMinZ = Infinity;
    let supportMaxX = -Infinity;
    let supportMaxY = -Infinity;
    let supportMaxZ = -Infinity;
    let candidateMinX = Infinity;
    let candidateMinY = Infinity;
    let candidateMinZ = Infinity;
    let candidateMaxX = -Infinity;
    let candidateMaxY = -Infinity;
    let candidateMaxZ = -Infinity;
    const scaleSamples: number[] = [];

    const readRadius = (index: number) => {
        if (!scales) return NaN;
        const logRadius = Math.max(scales[0][index], scales[1][index], scales[2][index]);
        return Number.isFinite(logRadius) ? Math.exp(Math.max(-30, Math.min(30, logRadius))) : NaN;
    };

    for (let index = 0; index < count; index++) {
        if (candidates[index] === 0 && support[index] === 0 && !silhouetteCore?.[index]) continue;
        const offset = index * 3;
        const x = centers[offset];
        const y = centers[offset + 1];
        const z = centers[offset + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
        candidateCount++;
        candidateMinX = Math.min(candidateMinX, x);
        candidateMinY = Math.min(candidateMinY, y);
        candidateMinZ = Math.min(candidateMinZ, z);
        candidateMaxX = Math.max(candidateMaxX, x);
        candidateMaxY = Math.max(candidateMaxY, y);
        candidateMaxZ = Math.max(candidateMaxZ, z);
        if (support[index] === 0 && !silhouetteCore?.[index]) continue;

        supportedCount++;
        supportMinX = Math.min(supportMinX, x);
        supportMinY = Math.min(supportMinY, y);
        supportMinZ = Math.min(supportMinZ, z);
        supportMaxX = Math.max(supportMaxX, x);
        supportMaxY = Math.max(supportMaxY, y);
        supportMaxZ = Math.max(supportMaxZ, z);
        const distanceSq = (x - anchor.x) ** 2 + (y - anchor.y) ** 2 + (z - anchor.z) ** 2;
        if (distanceSq < seedDistanceSq) {
            seedDistanceSq = distanceSq;
            seed = index;
        }
        if (scaleSamples.length < connectivityScaleSampleLimit) {
            const radius = readRadius(index);
            if (Number.isFinite(radius) && radius > 0) scaleSamples.push(radius);
        }
    }

    if (candidateCount === 0 || supportedCount === 0 || seed < 0) {
        return { hits: new Uint8Array(count), degraded: true };
    }

    let referenceScale: number;
    if (scaleSamples.length > 0) {
        scaleSamples.sort((a, b) => a - b);
        referenceScale = scaleSamples[Math.floor(scaleSamples.length / 2)];
    } else {
        const dx = candidateMaxX - candidateMinX;
        const dy = candidateMaxY - candidateMinY;
        const dz = candidateMaxZ - candidateMinZ;
        referenceScale = Math.hypot(dx, dy, dz) / Math.max(1, Math.cbrt(candidateCount) * 2);
    }
    if (!Number.isFinite(referenceScale) || referenceScale <= 0) {
        const strict = new Uint8Array(count);
        for (let index = 0; index < count; index++) {
            if (support[index] !== 0) strict[index] = 255;
        }
        return { hits: strict, degraded: true };
    }

    const supportSpanX = supportMaxX - supportMinX;
    const supportSpanY = supportMaxY - supportMinY;
    const supportSpanZ = supportMaxZ - supportMinZ;
    const largestSupportSpan = Math.max(supportSpanX, supportSpanY, supportSpanZ);
    const minimumPadding = referenceScale * connectivityEnvelopeMinimumScales;
    const paddingX = Math.max(minimumPadding, supportSpanX * connectivityEnvelopeRatio, largestSupportSpan * 0.05);
    const paddingY = Math.max(minimumPadding, supportSpanY * connectivityEnvelopeRatio, largestSupportSpan * 0.05);
    const paddingZ = Math.max(minimumPadding, supportSpanZ * connectivityEnvelopeRatio, largestSupportSpan * 0.05);
    const envelopeMinX = supportMinX - paddingX;
    const envelopeMinY = supportMinY - paddingY;
    const envelopeMinZ = supportMinZ - paddingZ;
    const envelopeMaxX = supportMaxX + paddingX;
    const envelopeMaxY = supportMaxY + paddingY;
    const envelopeMaxZ = supportMaxZ + paddingZ;

    const maximumDistance = referenceScale * connectivityMaximumDistanceFactor;
    const cellSize = maximumDistance;
    const eligibleIds = new Int32Array(candidateCount);
    const next = new Int32Array(candidateCount);
    next.fill(-1);
    const heads = new Map<string, number>();
    let eligibleCount = 0;
    let seedLocal = -1;
    const cellKey = (x: number, y: number, z: number) => {
        return `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)},${Math.floor(z / cellSize)}`;
    };

    for (let index = 0; index < count; index++) {
        if (candidates[index] === 0 && support[index] === 0 && !silhouetteCore?.[index]) continue;
        const offset = index * 3;
        const x = centers[offset];
        const y = centers[offset + 1];
        const z = centers[offset + 2];
        if (
            !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) ||
            x < envelopeMinX || x > envelopeMaxX ||
            y < envelopeMinY || y > envelopeMaxY ||
            z < envelopeMinZ || z > envelopeMaxZ
        ) {
            continue;
        }
        const local = eligibleCount++;
        eligibleIds[local] = index;
        if (index === seed) seedLocal = local;
        const key = cellKey(x, y, z);
        next[local] = heads.get(key) ?? -1;
        heads.set(key, local);
    }

    if (seedLocal < 0) {
        const strict = new Uint8Array(count);
        for (let index = 0; index < count; index++) {
            if (support[index] !== 0) strict[index] = 255;
        }
        return { hits: strict, degraded: true };
    }

    const clampedRadius = (index: number) => {
        const radius = readRadius(index);
        return Number.isFinite(radius) ?
            Math.max(referenceScale * 0.25, Math.min(referenceScale * 4, radius)) :
            referenceScale;
    };
    const colorsAreCompatible = (a: number, b: number) => {
        if (!colors) return true;
        const dr = (colors[0][a] - colors[0][b]) * SH_C0;
        const dg = (colors[1][a] - colors[1][b]) * SH_C0;
        const db = (colors[2][a] - colors[2][b]) * SH_C0;
        return !Number.isFinite(dr + dg + db) ||
            dr * dr + dg * dg + db * db <= connectivityColorDistance * connectivityColorDistance;
    };
    const areConnected = (a: number, b: number) => {
        const radiusA = clampedRadius(a);
        const radiusB = clampedRadius(b);
        if (Math.max(radiusA, radiusB) / Math.min(radiusA, radiusB) > connectivityMaximumScaleRatio) return false;
        const distance = Math.min(
            maximumDistance,
            Math.max(referenceScale * connectivityBaseDistanceFactor, (radiusA + radiusB) * 1.5)
        );
        const aOffset = a * 3;
        const bOffset = b * 3;
        const dx = centers[aOffset] - centers[bOffset];
        const dy = centers[aOffset + 1] - centers[bOffset + 1];
        const dz = centers[aOffset + 2] - centers[bOffset + 2];
        return dx * dx + dy * dy + dz * dz <= distance * distance && colorsAreCompatible(a, b);
    };

    const visited = new Uint8Array(eligibleCount);
    const depth = new Uint8Array(eligibleCount);
    const queue = new Int32Array(eligibleCount);
    let read = 0;
    let write = 0;
    const enqueueConnectedNeighbors = (local: number, trustedOnly: boolean) => {
        const index = eligibleIds[local];
        const offset = index * 3;
        const cellX = Math.floor(centers[offset] / cellSize);
        const cellY = Math.floor(centers[offset + 1] / cellSize);
        const cellZ = Math.floor(centers[offset + 2] / cellSize);
        for (let z = cellZ - 1; z <= cellZ + 1; z++) {
            for (let y = cellY - 1; y <= cellY + 1; y++) {
                for (let x = cellX - 1; x <= cellX + 1; x++) {
                    let neighbor = heads.get(`${x},${y},${z}`) ?? -1;
                    while (neighbor >= 0) {
                        const neighborIndex = eligibleIds[neighbor];
                        if (
                            neighbor !== local && visited[neighbor] === 0 &&
                            (!trustedOnly || support[neighborIndex] !== 0 || silhouetteCore?.[neighborIndex]) &&
                            areConnected(index, neighborIndex)
                        ) {
                            visited[neighbor] = 1;
                            if (!trustedOnly) depth[neighbor] = depth[local] + 1;
                            queue[write++] = neighbor;
                        }
                        neighbor = next[neighbor];
                    }
                }
            }
        }
    };

    const hasSilhouetteCore = silhouetteCore?.some(value => value !== 0) ?? false;
    if (hasSilhouetteCore) {
        visited[seedLocal] = 1;
        queue[write++] = seedLocal;
        while (read < write) {
            enqueueConnectedNeighbors(queue[read++], true);
        }
    } else {
        for (let local = 0; local < eligibleCount; local++) {
            if (support[eligibleIds[local]] === 0) continue;
            queue[write++] = local;
            visited[local] = 1;
        }
    }

    const trustedCount = write;
    read = 0;
    while (read < write) {
        const local = queue[read++];
        if (depth[local] >= connectivityMaximumGrowthHops) continue;
        enqueueConnectedNeighbors(local, false);
    }

    const reliable = write <= trustedCount * connectivityMaximumGrowthRatio;
    const result = new Uint8Array(count);
    if (reliable) {
        for (let local = 0; local < eligibleCount; local++) {
            if (visited[local] !== 0) result[eligibleIds[local]] = 255;
        }
        return { hits: result, degraded: false };
    }

    // Explosive local growth is ambiguous. Returning only positively observed
    // surfaces is safer than restoring the original depth-column candidates,
    // which would reintroduce hidden background.
    if (hasSilhouetteCore) {
        for (let local = 0; local < eligibleCount; local++) {
            const index = eligibleIds[local];
            if (visited[local] !== 0 && (support[index] !== 0 || silhouetteCore?.[index])) result[index] = 255;
        }
    } else {
        for (let index = 0; index < count; index++) {
            if (support[index] !== 0) result[index] = 255;
        }
    }
    return { hits: result, degraded: true };
};

const applyRefineMask = (hits: Uint8Array, state: Uint8Array, selectedBit: number) => {
    if (hits.length !== state.length) throw new Error('Selection and SAM2 masks have different sizes');
    const result = new Uint8Array(hits.length);
    for (let index = 0; index < hits.length; index++) {
        if (hits[index] === 255 && (state[index] & selectedBit) !== 0) result[index] = 255;
    }
    return result;
};

export {
    accumulateSegmentObservation,
    applyRefineMask,
    carveVisibleOutside,
    chooseStablePromptSample,
    createSegmentEvidenceAccumulator,
    evaluateAuxiliaryViewSafety,
    evaluateSegmentMaskQuality,
    finalizeSegmentEvidence,
    fuseSegmentObservations,
    selectSafestAuxiliaryPose,
    collectVisibilitySupport,
    filterAnchorConnectedSelection,
    findMaskInteriorPoint,
    hasMaskHits,
    projectMaskToSource,
    projectPredictionToSource,
    resolveCarvedSelection,
    sampleDistributedMaskPoints,
    sampleMaskPoints,
    segmentAuxiliaryMinimumAreaRatio,
    selectBestMask,
    suppressGroundPlaneLeak
};
export type {
    CarvedSelection,
    ConnectivitySelection,
    AuxiliaryViewRejectionReason,
    AuxiliaryViewSafety,
    AuxiliaryViewSafetyInput,
    AuxiliaryPoseCandidate,
    SegmentMaskQuality,
    SegmentMaskQualityInput,
    SegmentMaskRejectionReason,
    GaussianConnectivityData,
    GroundPlaneData,
    GroundPlaneSuppression,
    MaskCandidateSelectionOptions,
    Point3,
    SegmentOperation,
    SegmentPrediction,
    SegmentEvidenceAccumulator,
    SegmentFusion,
    SelectedMask,
    StablePromptData,
    StablePromptSample,
    VisibilityObservation
};
