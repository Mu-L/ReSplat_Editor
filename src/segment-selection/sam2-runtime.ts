import type * as Ort from 'onnxruntime-web/webgpu';

import {
    projectPredictionToSource,
    selectBestMask,
    type MaskCandidateSelectionOptions,
    type SegmentPrediction
} from './mask-utils';
import { SAM2_INPUT_SIZE, SAM2_MASK_INPUT_SIZE, SAM2_TINY_MODEL, type Sam2ModelFile } from './sam2-config';
import {
    normalizedPointsToModel,
    prepareSam2Image,
    sam2FrameSignature,
    type NormalizedPoint,
    type PreparedSam2Image
} from './sam2-image';
import { patchSam2EncoderForOrtWeb } from './sam2-model';

type OrtModule = typeof import('onnxruntime-web/webgpu');
type Sam2Backend = 'webgpu' | 'wasm';
type Sam2Stage = 'idle' | 'loading' | 'ready' | 'segmenting' | 'error';

type Sam2RuntimeStatus = {
    stage: Sam2Stage;
    backend?: Sam2Backend;
    file?: 'encoder' | 'decoder';
    loaded?: number;
    total?: number;
    message?: string;
};

type Sam2StatusListener = (status: Sam2RuntimeStatus) => void;
type Sam2Prompt = NormalizedPoint | readonly NormalizedPoint[];
type Sam2SegmentOptions = MaskCandidateSelectionOptions;

type EmbeddingCache = {
    key: string;
    outputs: Record<string, Ort.Tensor>;
};

const bytesToHex = (bytes: Uint8Array) => Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('');

class Sam2Runtime {
    private ort: OrtModule | null = null;
    private encoder: Ort.InferenceSession | null = null;
    private decoder: Ort.InferenceSession | null = null;
    private backend: Sam2Backend | null = null;
    private loadPromise: Promise<void> | null = null;
    private embedding: EmbeddingCache | null = null;
    private readonly onStatus: Sam2StatusListener;

    constructor(onStatus: Sam2StatusListener = () => {}) {
        this.onStatus = onStatus;
        this.emit({ stage: 'idle' });
    }

    get activeBackend() {
        return this.backend;
    }

    private emit(status: Sam2RuntimeStatus) {
        this.onStatus({ ...status, backend: status.backend ?? this.backend ?? undefined });
    }

    private async getOrt() {
        if (this.ort) return this.ort;
        const ort = await import('onnxruntime-web/webgpu');
        ort.env.wasm.wasmPaths = new URL('static/lib/onnxruntime/', document.baseURI).toString();
        ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
        this.ort = ort;
        return ort;
    }

    private async verifyModel(file: Sam2ModelFile, bytes: Uint8Array) {
        if (file.expectedBytes !== undefined && bytes.byteLength !== file.expectedBytes) {
            throw new Error(`${file.label} model size mismatch: expected ${file.expectedBytes}, received ${bytes.byteLength}`);
        }
        if (!globalThis.crypto?.subtle) return;
        // All model reads above create an ArrayBuffer-backed view. Narrowing the
        // generic typed-array buffer type keeps TypeScript 6's BufferSource
        // overload precise without another 150 MB copy.
        const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>);
        const actual = bytesToHex(new Uint8Array(digest));
        if (actual !== file.sha256) {
            throw new Error(`${file.label} model checksum mismatch`);
        }
    }

    private async fetchModel(file: Sam2ModelFile) {
        this.emit({ stage: 'loading', file: file.label });
        const url = new URL(file.url, document.baseURI);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load bundled ${file.label} model (${response.status})`);

        const bytes = new Uint8Array(await response.arrayBuffer());
        await this.verifyModel(file, bytes);
        return bytes;
    }

    private async releaseSessions() {
        await Promise.allSettled([this.encoder?.release(), this.decoder?.release()].filter(Boolean));
        this.encoder = null;
        this.decoder = null;
    }

    private async createSessions(backend: Sam2Backend, encoderBytes: Uint8Array, decoderBytes: Uint8Array) {
        const ort = await this.getOrt();
        const options: Ort.InferenceSession.SessionOptions = {
            executionProviders: [backend],
            graphOptimizationLevel: 'all'
        };
        this.emit({ stage: 'loading', backend, file: 'encoder' });
        this.encoder = await ort.InferenceSession.create(encoderBytes, options);
        try {
            this.emit({ stage: 'loading', backend, file: 'decoder' });
            this.decoder = await ort.InferenceSession.create(decoderBytes, options);
        } catch (error) {
            await this.encoder.release();
            this.encoder = null;
            throw error;
        }
        this.backend = backend;
    }

    load() {
        if (this.encoder && this.decoder) return;
        if (this.loadPromise) return this.loadPromise;

        this.loadPromise = (async () => {
            try {
                await this.getOrt();
                const encoderBytes = patchSam2EncoderForOrtWeb(await this.fetchModel(SAM2_TINY_MODEL.files[0]));
                const decoderBytes = await this.fetchModel(SAM2_TINY_MODEL.files[1]);
                const canUseWebGpu = 'gpu' in navigator;

                if (canUseWebGpu) {
                    try {
                        await this.createSessions('webgpu', encoderBytes, decoderBytes);
                    } catch (error) {
                        console.warn('[SAM2] WebGPU initialization failed; falling back to WASM', error);
                        await this.releaseSessions();
                    }
                }
                if (!this.encoder || !this.decoder) {
                    await this.createSessions('wasm', encoderBytes, decoderBytes);
                }
                this.emit({ stage: 'ready' });
            } catch (error) {
                this.emit({ stage: 'error', message: error instanceof Error ? error.message : String(error) });
                throw error;
            } finally {
                this.loadPromise = null;
            }
        })();
        return this.loadPromise;
    }

    private disposeEmbedding() {
        if (!this.embedding) return;
        for (const tensor of Object.values(this.embedding.outputs)) tensor.dispose();
        this.embedding = null;
    }

    private async encode(prepared: PreparedSam2Image, key: string) {
        await this.load();
        if (this.embedding?.key === key) return this.embedding.outputs;

        const inputName = this.encoder.inputNames[0];
        const image = new this.ort.Tensor('float32', prepared.tensorData, [1, 3, SAM2_INPUT_SIZE, SAM2_INPUT_SIZE]);
        let outputs: Record<string, Ort.Tensor>;
        try {
            outputs = await this.encoder.run({ [inputName]: image }) as Record<string, Ort.Tensor>;
        } finally {
            image.dispose();
        }
        this.disposeEmbedding();
        this.embedding = { key, outputs };
        return outputs;
    }

    private findEmbedding(outputs: Record<string, Ort.Tensor>, name: string) {
        const aliases: Record<string, string[]> = {
            image_embed: ['image_embed', 'image_embeddings'],
            image_embeddings: ['image_embeddings', 'image_embed'],
            high_res_feats_0: ['high_res_feats_0'],
            high_res_feats_1: ['high_res_feats_1']
        };
        for (const candidate of aliases[name] ?? [name]) {
            if (outputs[candidate]) return outputs[candidate];
        }
        return null;
    }

    async segment(
        rgba: Uint8Array,
        width: number,
        height: number,
        prompt: Sam2Prompt,
        options: Sam2SegmentOptions = {}
    ): Promise<SegmentPrediction> {
        this.emit({ stage: 'segmenting' });
        try {
            const prepared = prepareSam2Image(rgba, width, height);
            const embeddings = await this.encode(prepared, sam2FrameSignature(rgba, width, height));
            const points: readonly NormalizedPoint[] = Array.isArray(prompt) ? prompt : [prompt as NormalizedPoint];
            if (points.length === 0) throw new Error('SAM2 requires at least one positive prompt point');
            const pointCoordinates = normalizedPointsToModel(points, prepared.transform);
            const feeds: Record<string, Ort.Tensor> = {};
            const temporary: Ort.Tensor[] = [];

            for (const name of this.decoder.inputNames) {
                const embedding = this.findEmbedding(embeddings, name);
                if (embedding) {
                    feeds[name] = embedding;
                } else if (name === 'point_coords') {
                    feeds[name] = new this.ort.Tensor('float32', pointCoordinates, [1, points.length, 2]);
                    temporary.push(feeds[name]);
                } else if (name === 'point_labels') {
                    feeds[name] = new this.ort.Tensor('float32', new Float32Array(points.length).fill(1), [1, points.length]);
                    temporary.push(feeds[name]);
                } else if (name === 'mask_input') {
                    feeds[name] = new this.ort.Tensor('float32', new Float32Array(SAM2_MASK_INPUT_SIZE * SAM2_MASK_INPUT_SIZE), [1, 1, SAM2_MASK_INPUT_SIZE, SAM2_MASK_INPUT_SIZE]);
                    temporary.push(feeds[name]);
                } else if (name === 'has_mask_input') {
                    feeds[name] = new this.ort.Tensor('float32', new Float32Array([0]), [1]);
                    temporary.push(feeds[name]);
                } else if (name === 'orig_im_size' || name === 'img_size') {
                    feeds[name] = new this.ort.Tensor('float32', new Float32Array([SAM2_INPUT_SIZE, SAM2_INPUT_SIZE]), [2]);
                    temporary.push(feeds[name]);
                } else {
                    throw new Error(`Unsupported SAM2 decoder input: ${name}`);
                }
            }

            let outputs: Record<string, Ort.Tensor> | null = null;
            try {
                outputs = await this.decoder.run(feeds) as Record<string, Ort.Tensor>;
                const maskTensor = outputs.masks ?? outputs.pred_mask ?? outputs.low_res_masks ?? outputs[this.decoder.outputNames[0]];
                const scoreTensor = outputs.iou_predictions ?? outputs.iou ?? outputs[this.decoder.outputNames[1]];
                if (!maskTensor || !scoreTensor) throw new Error('SAM2 decoder did not return masks and IoU scores');
                const logits = new Float32Array(maskTensor.data as Float32Array);
                const scores = new Float32Array(scoreTensor.data as Float32Array);
                const best = selectBestMask(logits, maskTensor.dims, scores, options);
                return projectPredictionToSource(best, prepared.transform);
            } finally {
                temporary.forEach(tensor => tensor.dispose());
                if (outputs) Object.values(outputs).forEach(tensor => tensor.dispose());
            }
        } finally {
            if (this.encoder && this.decoder) this.emit({ stage: 'ready' });
        }
    }

    async dispose() {
        this.disposeEmbedding();
        await this.releaseSessions();
        this.backend = null;
        this.emit({ stage: 'idle' });
    }
}

export { Sam2Runtime };
export type { Sam2Backend, Sam2Prompt, Sam2RuntimeStatus, Sam2SegmentOptions, Sam2Stage, Sam2StatusListener };
