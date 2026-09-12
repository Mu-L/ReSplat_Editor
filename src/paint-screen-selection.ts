type NormalizedScreenPoint = { x: number; y: number };

// Allow a small amount behind the composited front-surface depth so the
// Gaussians which form that surface remain paintable. Candidates nearer than
// that depth are always accepted; only a second surface farther behind is
// rejected. Expressing the allowance as part of the brush footprint keeps it
// stable as the camera zooms.
const PAINT_FRONT_DEPTH_TOLERANCE_RATIO = 0.1;

// Give the brush a slightly more opaque, paint-like response without changing
// the meaning of the strength endpoints. A 0.6 UI strength becomes roughly
// 0.75 effective coverage, while 0 remains transparent and 1 remains opaque.
const PAINT_COVERAGE_POWER = 1.5;

const screenPaintCoverageStrength = (strength: number) => {
    const clamped = Math.min(1, Math.max(0, strength));
    return 1 - Math.pow(1 - clamped, PAINT_COVERAGE_POWER);
};

const paintFrontDepthTolerance = (worldBrushRadius: number) => (
    Math.max(Math.abs(worldBrushRadius) * PAINT_FRONT_DEPTH_TOLERANCE_RATIO, 1e-8)
);

const passesPaintFrontDepthLimit = (candidateDepth: number, frontDepth: number, tolerance: number) => (
    Number.isFinite(candidateDepth) &&
    Number.isFinite(frontDepth) &&
    candidateDepth - frontDepth <= Math.max(0, tolerance)
);

const screenPaintInterpolationSteps = (
    from: NormalizedScreenPoint,
    to: NormalizedScreenPoint,
    radiusPixels: number,
    viewportWidth: number,
    viewportHeight: number
) => {
    const dx = (to.x - from.x) * Math.max(1, viewportWidth);
    const dy = (to.y - from.y) * Math.max(1, viewportHeight);
    const distancePixels = Math.hypot(dx, dy);
    return Math.max(1, Math.ceil(distancePixels / Math.max(Math.abs(radiusPixels) * 0.5, 1e-8)));
};

export {
    PAINT_COVERAGE_POWER,
    PAINT_FRONT_DEPTH_TOLERANCE_RATIO,
    paintFrontDepthTolerance,
    passesPaintFrontDepthLimit,
    screenPaintCoverageStrength,
    screenPaintInterpolationSteps
};
export type { NormalizedScreenPoint };
