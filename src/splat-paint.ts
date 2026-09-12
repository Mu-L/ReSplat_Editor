import {
    BLENDEQUATION_MAX,
    BLENDMODE_ONE,
    BlendState,
    Color,
    GSPLAT_STREAM_INSTANCE,
    GSplatProcessor,
    GSplatResource,
    Mat4,
    PIXELFORMAT_RGBA8,
    Texture
} from 'playcanvas';

import { dcDecode, dcEncode } from './color-grade';
import type { Splat } from './splat';

const PAINT_STREAM = 'paintColor';

// Painting uses MAX blending so repeated samples in the same stroke do not
// accidentally compound their strength. The stroke uses one fixed color and
// strength, so MAX also preserves every texel written by earlier samples.
const paintBlendState = new BlendState(
    true,
    BLENDEQUATION_MAX,
    BLENDMODE_ONE,
    BLENDMODE_ONE,
    BLENDEQUATION_MAX,
    BLENDMODE_ONE,
    BLENDMODE_ONE
);

const processGLSL = /* glsl */ `
uniform vec4 uPaintScreenCircle;
uniform vec4 uPaintViewportDepth;
uniform vec4 uPaintColor;
uniform float uPaintHardness;
uniform float uPaintDepthUvFlip;
uniform mat4 uPaintModelView;
uniform mat4 uPaintModelViewProjection;
uniform highp sampler2D uPaintFrontDepth;
uniform highp sampler2D splatState;
uniform highp sampler2D soloMask;
uniform highp usampler2D splatTransform;
uniform highp sampler2D transformPalette;

vec3 getPaintCenter() {
    vec3 center = getCenter();
    uint transformIndex = texelFetch(splatTransform, splat.uv, 0).r;
    if (transformIndex == 0u) {
        return center;
    }

    int u = int(transformIndex % 512u) * 3;
    int v = int(transformIndex / 512u);
    mat4 transform;
    transform[0] = texelFetch(transformPalette, ivec2(u, v), 0);
    transform[1] = texelFetch(transformPalette, ivec2(u + 1, v), 0);
    transform[2] = texelFetch(transformPalette, ivec2(u + 2, v), 0);
    transform[3] = vec4(0.0, 0.0, 0.0, 1.0);
    return (transpose(transform) * vec4(center, 1.0)).xyz;
}

float getScreenBrushDistance(vec3 center) {
    vec4 clip = uPaintModelViewProjection * vec4(center, 1.0);
    if (clip.w <= 0.0) {
        return -1.0;
    }

    vec2 screen = vec2(
        clip.x / clip.w * 0.5 + 0.5,
        1.0 - (clip.y / clip.w * 0.5 + 0.5)
    );
    if (any(lessThan(screen, vec2(0.0))) || any(greaterThanEqual(screen, vec2(1.0)))) {
        return -1.0;
    }

    float distancePixels = length((screen - uPaintScreenCircle.xy) * uPaintViewportDepth.xy);
    if (distancePixels >= uPaintScreenCircle.z) {
        return -1.0;
    }

    ivec2 depthSize = textureSize(uPaintFrontDepth, 0);
    vec2 depthUv = screen;
    if (uPaintDepthUvFlip > 0.5) {
        depthUv.y = 1.0 - depthUv.y;
    }
    ivec2 depthPixel = clamp(
        ivec2(floor(depthUv * vec2(depthSize))),
        ivec2(0),
        depthSize - ivec2(1)
    );
    vec4 packedFrontDepth = texelFetch(uPaintFrontDepth, depthPixel, 0);
    float frontAlpha = 1.0 - packedFrontDepth.a;
    if (frontAlpha < 1e-6) {
        return -1.0;
    }

    float frontNormalizedDepth = clamp(packedFrontDepth.r / frontAlpha, 0.0, 1.0);
    float frontLinearDepth = mix(uPaintViewportDepth.z, uPaintViewportDepth.w, frontNormalizedDepth);
    float linearDepth = -(uPaintModelView * vec4(center, 1.0)).z;
    if (linearDepth - frontLinearDepth > uPaintScreenCircle.w) {
        return -1.0;
    }

    return distancePixels;
}

void process() {
    vec4 result = vec4(0.0);
    uint vertexState = uint(texelFetch(splatState, splat.uv, 0).r * 255.0 + 0.5) & 7u;
    bool editable = (vertexState & 6u) == 0u;
    bool visible = texelFetch(soloMask, splat.uv, 0).r >= 0.5;

    if (editable && visible) {
        vec3 center = getPaintCenter();
        float radius = uPaintScreenCircle.z;
        float distanceToCenter = getScreenBrushDistance(center);
        if (distanceToCenter >= 0.0 && distanceToCenter < radius) {
            float hardness = clamp(uPaintHardness, 0.0, 1.0);
            float falloff = hardness >= 0.999
                ? 1.0
                : 1.0 - smoothstep(radius * hardness, radius, distanceToCenter);
            result = uPaintColor;
            result.a *= falloff;
        }
    }

    writePaintColor(result);
}
`;

type PaintSettings = {
    color: Color;
    strength: number;
    hardness: number;
    radius: number;
};

type ScreenPaintSettings = Omit<PaintSettings, 'radius'> & {
    x: number;
    y: number;
    radiusPixels: number;
    viewportWidth: number;
    viewportHeight: number;
    depthTolerance: number;
    nearClip: number;
    farClip: number;
    frontDepthTexture: Texture;
    viewMatrix: Mat4;
    viewProjectionMatrix: Mat4;
};

type PaintStrokeDelta = {
    indices: Uint32Array;
    before: Float32Array;
    after: Float32Array;
    colors: Float32Array;
    beforeShMask: Uint8Array;
    afterShMask: Uint8Array;
};

type PaintCommitOptions = {
    collectDiagnostic: true;
};

type PaintCommitDiagnostic = {
    sphereIndices: Uint32Array;
    frontVisibleIndices: Uint32Array;
    paintThroughIndices: Uint32Array;
    timings: {
        gpuReadbackMs: number;
        commitCpuMs: number;
        totalCommitMs: number;
    };
};

type PaintDiagnosticCommitResult = {
    delta: PaintStrokeDelta | null;
    diagnostic: PaintCommitDiagnostic;
};

type PaintEraseDelta = {
    indices: Uint32Array;
    strengths: Float32Array;
};

type PaintSampleData = {
    indices: Uint32Array;
    colors: Float32Array;
};

type PaintErasePreview = {
    indices: Uint32Array;
    colors: Float32Array;
    shMask: Uint8Array;
};

type PaintPreviewMode = 'paint' | 'erase-color' | 'erase-opacity';

type PaintEraseCommitOptions = {
    keepPreview?: boolean;
};

class SplatPaintRuntime {
    readonly splat: Splat;
    readonly texture: Texture;
    readonly eraseTargetTexture: Texture;

    private processor: GSplatProcessor;
    private screenCircle = new Float32Array(4);
    private viewportDepth = new Float32Array(4);
    private color = new Float32Array(4);
    private modelView = new Mat4();
    private modelViewProjection = new Mat4();
    private previewMode: PaintPreviewMode = 'paint';
    private previewEnabled = true;
    private destroyed = false;

    constructor(splat: Splat) {
        this.splat = splat;

        const resource = splat.asset.resource as GSplatResource;
        if (!resource.format.getStream(PAINT_STREAM)) {
            resource.format.addExtraStreams([{
                name: PAINT_STREAM,
                format: PIXELFORMAT_RGBA8,
                storage: GSPLAT_STREAM_INSTANCE
            }]);
        }

        // The stroke overlay belongs to this Splat instance, not to its backing
        // resource. Multiple scene objects are allowed to reference one resource
        // (for example after duplication); a resource texture would make every
        // such object sample the stroke while the pointer is held down.
        const { x: width, y: height } = resource.textureDimensions;
        const texture = Texture.createDataTexture2D(
            resource.device,
            `${PAINT_STREAM}-${splat.uid}`,
            width,
            height,
            PIXELFORMAT_RGBA8
        );
        this.texture = texture;
        this.eraseTargetTexture = Texture.createDataTexture2D(
            resource.device,
            `paintEraseTarget-${splat.uid}`,
            width,
            height,
            PIXELFORMAT_RGBA8
        );
        const eraseTargetData = this.eraseTargetTexture.lock() as Uint8Array;
        eraseTargetData.fill(0);
        this.eraseTargetTexture.unlock();
        this.clear();

        // GSplatProcessor resolves instance streams through a component binding.
        // Legacy rendering does not create engine-owned instance textures, so
        // provide the runtime-owned texture explicitly while retaining the real
        // resource for stream layout and Gaussian count information.
        const paintDestination = {
            resource,
            getInstanceTexture: (name: string) => (name === PAINT_STREAM ? texture : null)
        };

        this.processor = new GSplatProcessor(
            resource.device,
            { resource },
            { component: paintDestination as any, streams: [PAINT_STREAM] },
            { processGLSL }
        );
        this.processor.blendState = paintBlendState;
        this.processor.setParameter('splatState', splat.stateTexture);
        this.processor.setParameter('soloMask', splat.soloMaskTexture);
        this.processor.setParameter('splatTransform', splat.transformTexture);
        this.processor.setParameter('transformPalette', splat.transformPalette.texture);
        this.processor.setParameter('uPaintScreenCircle', this.screenCircle);
        this.processor.setParameter('uPaintViewportDepth', this.viewportDepth);
        this.processor.setParameter('uPaintColor', this.color);
        this.processor.setParameter('uPaintHardness', 1);
        this.processor.setParameter('uPaintDepthUvFlip', resource.device.isWebGPU ? 0 : 1);
        this.processor.setParameter('uPaintModelView', this.modelView.data);
        this.processor.setParameter('uPaintModelViewProjection', this.modelViewProjection.data);
        this.processor.setParameter('uPaintFrontDepth', texture);

        this.bindPreview();
    }

    private bindPreview() {
        const mode = this.previewMode === 'paint' ? 0 : (this.previewMode === 'erase-color' ? 1 : 2);
        this.splat.setPaintTexture(
            this.previewEnabled ? this.texture : null,
            mode,
            mode === 1 ? this.eraseTargetTexture : null
        );
    }

    setPreviewEnabled(enabled: boolean) {
        if (this.destroyed || this.previewEnabled === enabled) return;
        this.previewEnabled = enabled;
        this.bindPreview();
        if (this.splat.scene) this.splat.scene.forceRender = true;
    }

    setPreviewMode(mode: PaintPreviewMode) {
        if (this.destroyed || this.previewMode === mode) return;
        this.previewMode = mode;
        this.bindPreview();
    }

    setErasePreviewTarget(preview: PaintErasePreview | null | undefined) {
        if (this.destroyed) return;

        const count = this.splat.splatData.numSplats;
        const dc0 = this.splat.splatData.getProp('f_dc_0') as Float32Array;
        const dc1 = this.splat.splatData.getProp('f_dc_1') as Float32Array;
        const dc2 = this.splat.splatData.getProp('f_dc_2') as Float32Array;
        const data = this.eraseTargetTexture.lock() as Uint8Array;
        data.fill(0);
        for (let index = 0; index < count; ++index) {
            const pixel = index * 4;
            data[pixel] = Math.round(Math.min(1, Math.max(0, dcDecode(dc0[index]))) * 255);
            data[pixel + 1] = Math.round(Math.min(1, Math.max(0, dcDecode(dc1[index]))) * 255);
            data[pixel + 2] = Math.round(Math.min(1, Math.max(0, dcDecode(dc2[index]))) * 255);
            data[pixel + 3] = this.splat.paintShMaskData[index];
        }
        if (preview) {
            if (preview.colors.length !== preview.indices.length * 3 || preview.shMask.length !== preview.indices.length) {
                this.eraseTargetTexture.unlock();
                throw new Error('Paint erase preview data does not match its Gaussian indices.');
            }
            for (let i = 0; i < preview.indices.length; ++i) {
                const index = preview.indices[i];
                if (index >= count) continue;
                const pixel = index * 4;
                const value = i * 3;
                data[pixel] = Math.round(Math.min(1, Math.max(0, preview.colors[value])) * 255);
                data[pixel + 1] = Math.round(Math.min(1, Math.max(0, preview.colors[value + 1])) * 255);
                data[pixel + 2] = Math.round(Math.min(1, Math.max(0, preview.colors[value + 2])) * 255);
                data[pixel + 3] = preview.shMask[i];
            }
        }
        this.eraseTargetTexture.unlock();
        if (this.splat.scene) this.splat.scene.forceRender = true;
    }

    paintScreenCircle(settings: ScreenPaintSettings) {
        if (this.destroyed || !this.splat.scene) return;

        this.screenCircle[0] = Math.min(1, Math.max(0, settings.x));
        this.screenCircle[1] = Math.min(1, Math.max(0, settings.y));
        this.screenCircle[2] = Math.max(settings.radiusPixels, 1e-8);
        this.screenCircle[3] = Math.max(settings.depthTolerance, 0);
        this.viewportDepth[0] = Math.max(settings.viewportWidth, 1);
        this.viewportDepth[1] = Math.max(settings.viewportHeight, 1);
        this.viewportDepth[2] = settings.nearClip;
        this.viewportDepth[3] = settings.farClip;
        this.color[0] = settings.color.r;
        this.color[1] = settings.color.g;
        this.color[2] = settings.color.b;
        this.color[3] = Math.min(1, Math.max(0, settings.strength));
        this.modelView.mul2(settings.viewMatrix, this.splat.worldTransform);
        this.modelViewProjection.mul2(settings.viewProjectionMatrix, this.splat.worldTransform);
        this.processor.setParameter('uPaintHardness', Math.min(1, Math.max(0, settings.hardness)));
        this.processor.setParameter('uPaintFrontDepth', settings.frontDepthTexture);

        this.processor.process();
        this.splat.scene.forceRender = true;
    }

    paintSamples({ indices, colors }: PaintSampleData) {
        if (this.destroyed || !this.splat.scene) return;
        if (colors.length !== indices.length * 4) {
            throw new Error('Paint samples require four RGBA values per splat.');
        }

        const data = this.texture.lock() as Uint8Array;
        for (let i = 0; i < indices.length; ++i) {
            const splatIndex = indices[i];
            if (splatIndex >= this.splat.splatData.numSplats) continue;
            const src = i * 4;
            const dst = splatIndex * 4;
            data[dst] = Math.round(Math.min(1, Math.max(0, colors[src])) * 255);
            data[dst + 1] = Math.round(Math.min(1, Math.max(0, colors[src + 1])) * 255);
            data[dst + 2] = Math.round(Math.min(1, Math.max(0, colors[src + 2])) * 255);
            data[dst + 3] = Math.round(Math.min(1, Math.max(0, colors[src + 3])) * 255);
        }
        this.texture.unlock();
        this.splat.scene.forceRender = true;
    }

    private async readStrokePixels(collectDiagnostic = false) {
        if (this.destroyed || !this.splat.scene) {
            return null;
        }

        const resource = this.splat.asset.resource as GSplatResource;
        const { x: width, y: height } = resource.textureDimensions;
        const readbackStart = performance.now();
        const pixels = await this.texture.read(0, 0, width, height, {
            immediate: true
        }) as Uint8Array;
        const gpuReadbackMs = performance.now() - readbackStart;

        if (this.destroyed || !this.splat.scene) {
            return null;
        }

        const count = this.splat.splatData.numSplats;
        let sphereIndices = new Uint32Array(0);
        const frontVisibleIndices = new Uint32Array(0);
        const paintThroughIndices = new Uint32Array(0);
        if (collectDiagnostic) {
            let sphereCount = 0;
            for (let index = 0; index < count; index++) {
                if (pixels[index * 4 + 3] !== 0) sphereCount++;
            }
            sphereIndices = new Uint32Array(sphereCount);
            let write = 0;
            for (let index = 0; index < count; index++) {
                if (pixels[index * 4 + 3] !== 0) sphereIndices[write++] = index;
            }
        }
        return { pixels, sphereIndices, frontVisibleIndices, paintThroughIndices, gpuReadbackMs };
    }

    async commit(): Promise<PaintStrokeDelta | null>;
    async commit(options: PaintCommitOptions): Promise<PaintDiagnosticCommitResult>;
    async commit(options?: PaintCommitOptions): Promise<PaintStrokeDelta | PaintDiagnosticCommitResult | null> {
        const totalStart = performance.now();
        const collectDiagnostic = options?.collectDiagnostic === true;
        const read = await this.readStrokePixels(collectDiagnostic);
        if (!read) {
            if (!collectDiagnostic) return null;
            return {
                delta: null,
                diagnostic: {
                    sphereIndices: new Uint32Array(0),
                    frontVisibleIndices: new Uint32Array(0),
                    paintThroughIndices: new Uint32Array(0),
                    timings: {
                        gpuReadbackMs: 0,
                        commitCpuMs: 0,
                        totalCommitMs: performance.now() - totalStart
                    }
                }
            };
        }
        const { pixels, sphereIndices, frontVisibleIndices, paintThroughIndices, gpuReadbackMs } = read;

        const complete = (delta: PaintStrokeDelta | null): PaintStrokeDelta | PaintDiagnosticCommitResult | null => {
            if (!collectDiagnostic) return delta;
            const totalCommitMs = performance.now() - totalStart;
            return {
                delta,
                diagnostic: {
                    sphereIndices,
                    frontVisibleIndices,
                    paintThroughIndices,
                    timings: {
                        gpuReadbackMs,
                        commitCpuMs: Math.max(0, totalCommitMs - gpuReadbackMs),
                        totalCommitMs
                    }
                }
            };
        };

        const count = this.splat.splatData.numSplats;

        let changed = 0;
        for (let i = 0; i < count; ++i) {
            if (pixels[i * 4 + 3] !== 0) changed++;
        }

        if (changed === 0) {
            this.clear();
            return complete(null);
        }

        const indices = new Uint32Array(changed);
        const before = new Float32Array(changed * 3);
        const after = new Float32Array(changed * 3);
        const colors = new Float32Array(changed * 4);
        const beforeShMask = new Uint8Array(changed);
        const afterShMask = new Uint8Array(changed);
        const dc0 = this.splat.splatData.getProp('f_dc_0') as Float32Array;
        const dc1 = this.splat.splatData.getProp('f_dc_1') as Float32Array;
        const dc2 = this.splat.splatData.getProp('f_dc_2') as Float32Array;

        let dst = 0;
        for (let i = 0; i < count; ++i) {
            const pixel = i * 4;
            const strength = pixels[pixel + 3] / 255;
            if (strength === 0) continue;

            const value = dst * 3;
            indices[dst] = i;
            before[value] = dc0[i];
            before[value + 1] = dc1[i];
            before[value + 2] = dc2[i];

            const invStrength = 1 - strength;
            after[value] = dcEncode(dcDecode(dc0[i]) * invStrength + pixels[pixel] / 255 * strength);
            after[value + 1] = dcEncode(dcDecode(dc1[i]) * invStrength + pixels[pixel + 1] / 255 * strength);
            after[value + 2] = dcEncode(dcDecode(dc2[i]) * invStrength + pixels[pixel + 2] / 255 * strength);
            const color = dst * 4;
            colors[color] = pixels[pixel] / 255;
            colors[color + 1] = pixels[pixel + 1] / 255;
            colors[color + 2] = pixels[pixel + 2] / 255;
            colors[color + 3] = strength;
            beforeShMask[dst] = this.splat.paintShMaskData[i];
            afterShMask[dst] = Math.round((strength + beforeShMask[dst] / 255 * invStrength) * 255);
            dst++;
        }

        this.splat.applyPaintValues(indices, after);
        this.splat.applyPaintShMaskValues(indices, afterShMask);
        this.clear();
        return complete({ indices, before, after, colors, beforeShMask, afterShMask });
    }

    async commitErase(options: PaintEraseCommitOptions = {}): Promise<PaintEraseDelta | null> {
        const read = await this.readStrokePixels();
        if (!read) return null;
        const { pixels } = read;

        const count = this.splat.splatData.numSplats;
        let changed = 0;
        for (let i = 0; i < count; ++i) {
            if (pixels[i * 4 + 3] !== 0) changed++;
        }

        if (changed === 0) {
            if (!options.keepPreview) this.clear();
            return null;
        }

        const indices = new Uint32Array(changed);
        const strengths = new Float32Array(changed);
        let dst = 0;
        for (let i = 0; i < count; ++i) {
            const strength = pixels[i * 4 + 3] / 255;
            if (strength === 0) continue;
            indices[dst] = i;
            strengths[dst] = strength;
            dst++;
        }

        if (!options.keepPreview) this.clear();
        return { indices, strengths };
    }

    clear() {
        if (this.destroyed) return;
        const data = this.texture.lock() as Uint8Array;
        data.fill(0);
        this.texture.unlock();
        if (this.splat.scene) this.splat.scene.forceRender = true;
    }

    destroy() {
        if (this.destroyed) return;
        this.clear();
        this.destroyed = true;
        this.processor?.destroy();
        this.splat.setPaintTexture(null);
        this.texture.destroy();
        this.eraseTargetTexture.destroy();
    }
}

export { SplatPaintRuntime };
export type {
    PaintCommitDiagnostic,
    PaintDiagnosticCommitResult,
    PaintErasePreview,
    PaintEraseDelta,
    PaintPreviewMode,
    PaintSampleData,
    ScreenPaintSettings,
    PaintSettings,
    PaintStrokeDelta
};
