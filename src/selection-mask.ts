import type { Splat } from './splat';

type SelectionMaskOperation = 'add' | 'remove' | 'set';

type MaskProjection = 'auto' | 'visible' | 'centers';

type ResolveMaskOptions = {
    /**
     * `centers` tests every Gaussian center against the 2D mask instead of
     * resolving only the front-most rendered ID at each pixel. `visible`
     * forces front-most ID resolution even while the editor is displaying
     * centers or an overlay. `auto` preserves the legacy editor behaviour.
     */
    projection?: MaskProjection;
    /** Ignore Gaussian fragments whose effective alpha is below this value. */
    alphaThreshold?: number;
    /** Also return every reliable front-most Gaussian observed in the view. */
    collectVisible?: boolean;
    /** Segment Select's source-resolution confidence map for evidence fusion. */
    sourceConfidence?: Uint8Array;
    /** SAM2's predicted IoU for the selected mask candidate. */
    predictedIou?: number;
};

type ResolvedMaskSelection = {
    splat: Splat;
    /** Per-Gaussian committed snapshot. A value of 255 means hit. */
    hits: Uint8Array;
    /** Present when collectVisible is requested for visible projection. */
    visible?: Uint8Array;
    /** Number of visible source pixels accumulated for each Gaussian. */
    coverage?: Uint16Array;
    /** Mean source-mask confidence accumulated for each visible Gaussian. */
    confidence?: Uint8Array;
    /** SAM2's predicted IoU for this view. */
    predictedIou?: number;
};

export type { MaskProjection, ResolvedMaskSelection, ResolveMaskOptions, SelectionMaskOperation };
