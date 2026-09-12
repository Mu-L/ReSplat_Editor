import { SAM2_INPUT_SIZE } from './sam2-config';

type NormalizedPoint = { x: number; y: number };

type Sam2ImageTransform = {
    sourceWidth: number;
    sourceHeight: number;
    modelSize: number;
    scaledWidth: number;
    scaledHeight: number;
    scaleX: number;
    scaleY: number;
    padX: number;
    padY: number;
};

type PreparedSam2Image = {
    tensorData: Float32Array;
    transform: Sam2ImageTransform;
};

const MEAN = [0.485, 0.456, 0.406] as const;
const STD = [0.229, 0.224, 0.225] as const;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const prepareSam2Image = (
    rgba: Uint8Array,
    sourceWidth: number,
    sourceHeight: number,
    modelSize = SAM2_INPUT_SIZE
): PreparedSam2Image => {
    if (sourceWidth <= 0 || sourceHeight <= 0 || rgba.length !== sourceWidth * sourceHeight * 4) {
        throw new Error('Invalid RGBA image dimensions');
    }

    // The vietanhdev SAM2 export uses a direct 1024 x 1024 resize. Its decoder
    // likewise expects prompt coordinates scaled independently on each axis.
    const scaledWidth = modelSize;
    const scaledHeight = modelSize;
    const padX = 0;
    const padY = 0;
    const scaleX = scaledWidth / sourceWidth;
    const scaleY = scaledHeight / sourceHeight;
    const planeSize = modelSize * modelSize;
    const tensorData = new Float32Array(planeSize * 3);

    for (let dy = 0; dy < scaledHeight; dy++) {
        const sy = clamp((dy + 0.5) / scaleY - 0.5, 0, sourceHeight - 1);
        const y0 = Math.floor(sy);
        const y1 = Math.min(sourceHeight - 1, y0 + 1);
        const fy = sy - y0;

        for (let dx = 0; dx < scaledWidth; dx++) {
            const sx = clamp((dx + 0.5) / scaleX - 0.5, 0, sourceWidth - 1);
            const x0 = Math.floor(sx);
            const x1 = Math.min(sourceWidth - 1, x0 + 1);
            const fx = sx - x0;
            const p00 = (y0 * sourceWidth + x0) * 4;
            const p10 = (y0 * sourceWidth + x1) * 4;
            const p01 = (y1 * sourceWidth + x0) * 4;
            const p11 = (y1 * sourceWidth + x1) * 4;
            const target = (padY + dy) * modelSize + padX + dx;

            for (let channel = 0; channel < 3; channel++) {
                const top = rgba[p00 + channel] * (1 - fx) + rgba[p10 + channel] * fx;
                const bottom = rgba[p01 + channel] * (1 - fx) + rgba[p11 + channel] * fx;
                const value = (top * (1 - fy) + bottom * fy) / 255;
                tensorData[channel * planeSize + target] = (value - MEAN[channel]) / STD[channel];
            }
        }
    }

    return {
        tensorData,
        transform: {
            sourceWidth,
            sourceHeight,
            modelSize,
            scaledWidth,
            scaledHeight,
            scaleX,
            scaleY,
            padX,
            padY
        }
    };
};

const normalizedPointToModel = (point: NormalizedPoint, transform: Sam2ImageTransform): [number, number] => {
    return [
        clamp(point.x, 0, 1) * transform.sourceWidth * transform.scaleX + transform.padX,
        clamp(point.y, 0, 1) * transform.sourceHeight * transform.scaleY + transform.padY
    ];
};

const normalizedPointsToModel = (points: readonly NormalizedPoint[], transform: Sam2ImageTransform) => {
    const result = new Float32Array(points.length * 2);
    for (let index = 0; index < points.length; index++) {
        const [x, y] = normalizedPointToModel(points[index], transform);
        result[index * 2] = x;
        result[index * 2 + 1] = y;
    }
    return result;
};

// Hash every RGB pixel so an embedding is never reused after a scene change
// merely because the changed pixels fell between a set of sparse samples.
const sam2FrameSignature = (rgba: Uint8Array, width: number, height: number) => {
    let hash = 2166136261;
    for (let index = 0; index < rgba.length; index += 4) {
        hash ^= rgba[index];
        hash = Math.imul(hash, 16777619);
        hash ^= rgba[index + 1];
        hash = Math.imul(hash, 16777619);
        hash ^= rgba[index + 2];
        hash = Math.imul(hash, 16777619);
    }
    return `${width}x${height}:${(hash >>> 0).toString(16)}`;
};

export { normalizedPointToModel, normalizedPointsToModel, prepareSam2Image, sam2FrameSignature };
export type { NormalizedPoint, PreparedSam2Image, Sam2ImageTransform };
