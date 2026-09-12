import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import ts from 'typescript';

const sourceUrl = new URL('../src/paint-screen-selection.ts', import.meta.url);
const source = readFileSync(sourceUrl, 'utf8');
const compiled = ts.transpileModule(source, {
    compilerOptions: {
        module: ts.ModuleKind.ES2022,
        target: ts.ScriptTarget.ES2022
    }
}).outputText;
const moduleUrl = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`;
const {
    PAINT_COVERAGE_POWER,
    PAINT_FRONT_DEPTH_TOLERANCE_RATIO,
    paintFrontDepthTolerance,
    passesPaintFrontDepthLimit,
    screenPaintCoverageStrength,
    screenPaintInterpolationSteps
} = await import(moduleUrl);

assert.equal(PAINT_COVERAGE_POWER, 1.5);
assert.equal(PAINT_FRONT_DEPTH_TOLERANCE_RATIO, 0.1);
assert.equal(paintFrontDepthTolerance(2), 0.2);
assert.equal(paintFrontDepthTolerance(-2), 0.2);
assert.equal(passesPaintFrontDepthLimit(10.1, 10, 0.1), true);
assert.equal(passesPaintFrontDepthLimit(9.9, 10, 0.1), true);
assert.equal(passesPaintFrontDepthLimit(10.1001, 10, 0.1), false);
assert.equal(passesPaintFrontDepthLimit(1, 10, 0.1), true);
assert.equal(screenPaintCoverageStrength(0), 0);
assert.equal(screenPaintCoverageStrength(1), 1);
assert.equal(screenPaintCoverageStrength(-1), 0);
assert.equal(screenPaintCoverageStrength(2), 1);
assert.ok(Math.abs(screenPaintCoverageStrength(0.6) - 0.7470177871865296) < 1e-12);

// Regression values from ReSplat-Paint-Diagnostic-cat-girl: Gaussian 540062
// was 0.118386 world units behind the sampled front surface while the brush
// radius was 0.46208. The 10% depth band rejects it, while retaining a point
// whose center differs from the front-depth estimate only by 0.003043.
const catGirlTolerance = paintFrontDepthTolerance(0.46208);
assert.equal(passesPaintFrontDepthLimit(12.173338, 12.054952, catGirlTolerance), false);
assert.equal(passesPaintFrontDepthLimit(12.051909, 12.054952, catGirlTolerance), true);

assert.equal(screenPaintInterpolationSteps(
    { x: 0.1, y: 0.5 },
    { x: 0.3, y: 0.5 },
    40,
    1000,
    500
), 10);
assert.equal(screenPaintInterpolationSteps(
    { x: 0.5, y: 0.5 },
    { x: 0.5, y: 0.5 },
    40,
    1000,
    500
), 1);

const runtimeSource = readFileSync(new URL('../src/splat-paint.ts', import.meta.url), 'utf8');
assert.match(runtimeSource, /length\(\(screen - uPaintScreenCircle\.xy\) \* uPaintViewportDepth\.xy\)/);
assert.match(runtimeSource, /linearDepth - frontLinearDepth > uPaintScreenCircle\.w/);
assert.doesNotMatch(runtimeSource, /queuePrimaryPaintSample\(/);
assert.doesNotMatch(runtimeSource, /queuedPrimarySamples/);

const toolSource = readFileSync(new URL('../src/tools/paint-tool.ts', import.meta.url), 'utf8');
assert.match(toolSource, /runtime\.paintScreenCircle\(/);
assert.match(toolSource, /strength: screenPaintCoverageStrength\(settings\.strength\)/);
assert.doesNotMatch(toolSource, /queueFrontVisibleBrushSamples\(/);
assert.doesNotMatch(toolSource, /runtime\.queuePrimaryPaintSample\(/);
assert.doesNotMatch(toolSource, /runtime\.queuePaintSample\(/);
assert.match(toolSource, /if \(isBrushPaintTool\(strokeTool\)\)[\s\S]*runtime\.paintScreenCircle\(/);
assert.match(toolSource, /scene\.camera\.intersect\([\s\S]*paintPickAlphaThreshold,[\s\S]*eraseCandidates,[\s\S]*true/);
assert.match(toolSource, /setErasePreviewTarget\(erasePreview\)/);
assert.match(toolSource, /commitErase\(\{ keepPreview: true \}\)/);
assert.match(toolSource, /await operation\.do\(\);\s*events\.fire\('edit\.add', operation, true\)/);

const cameraSource = readFileSync(new URL('../src/camera.ts', import.meta.url), 'utf8');
assert.match(cameraSource, /if \(retainClosestDepthMap\) \{\s*this\.picker\.prepareDepth\(closestSplat, alphaThreshold\)/);
assert.doesNotMatch(cameraSource, /retainClosestDepthMap && lastPreparedSplat/);

const shaderSource = readFileSync(new URL('../src/shaders/splat-shader.ts', import.meta.url), 'utf8');
assert.match(shaderSource, /float shPaintFactor = 1\.0 - texelFetch\(paintShMask, splat\.uv, 0\)\.r/);
assert.match(shaderSource, /shPaintFactor \*= 1\.0 - painted\.a/);
assert.match(shaderSource, /evalSH\(sh, dir\) \* scale \* shPaintFactor/);
assert.match(shaderSource, /paintPreviewMode < 1\.5/);
assert.match(shaderSource, /color\.xyz = mix\(color\.xyz, eraseTarget\.xyz, painted\.a\)/);
assert.match(shaderSource, /color\.a \*= 1\.0 - painted\.a/);

const splatSource = readFileSync(new URL('../src/splat.ts', import.meta.url), 'utf8');
assert.match(splatSource, /applyPaintShMaskValues\(/);
assert.match(splatSource, /setParameter\('paintEraseTarget'/);

const paintLayersSource = readFileSync(new URL('../src/paint-layers.ts', import.meta.url), 'utf8');
assert.match(paintLayersSource, /paint\.layers\.erasePreview/);
assert.match(paintLayersSource, /createErasePreview\(/);
assert.match(paintLayersSource, /layer\.id === erasedLayerId/);

const editOpsSource = readFileSync(new URL('../src/edit-ops.ts', import.meta.url), 'utf8');
assert.match(editOpsSource, /beforeShMask/);
assert.match(editOpsSource, /afterShMask/);

console.log('paint screen-space front-surface regression test passed');
