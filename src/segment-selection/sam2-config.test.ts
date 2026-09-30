// @ts-expect-error Test-only Node builtin; browser builds do not import this module.
import { createHash } from 'node:crypto';
// @ts-expect-error Test-only Node builtin; browser builds do not import this module.
import { createReadStream, statSync } from 'node:fs';
// @ts-expect-error Test-only Node builtin; browser builds do not import this module.
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { getSam2ModelUrls, SAM2_TINY_MODEL } from './sam2-config';

const sha256File = (filePath: string) => new Promise<string>((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk: Uint8Array) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
});

describe('bundled SAM2 model files', () => {
    for (const file of SAM2_TINY_MODEL.files) {
        it(`ships the pinned ${file.label}`, async () => {
            expect(file.url).toMatch(/^static\/models\/sam2\/[^/]+\.onnx$/);
            expect(file.remoteUrl).toBe(
                `https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/${SAM2_TINY_MODEL.revision}/${file.url.split('/').at(-1)}`
            );

            const filePath = fileURLToPath(new URL(`../../${file.url}`, import.meta.url));
            expect(statSync(filePath).size).toBe(file.expectedBytes);
            await expect(sha256File(filePath)).resolves.toBe(file.sha256);
        });
    }
});

describe('SAM2 model URL priority', () => {
    const encoder = SAM2_TINY_MODEL.files[0];

    it('prefers the pinned remote model on GitHub Pages', () => {
        const urls = getSam2ModelUrls(encoder, 'https://re-qi.github.io/ReSplat_Editor/');
        expect(urls.map(url => url.toString())).toEqual([
            encoder.remoteUrl,
            'https://re-qi.github.io/ReSplat_Editor/static/models/sam2/sam2_hiera_tiny.encoder.onnx'
        ]);
    });

    it('keeps the below-limit decoder on GitHub Pages', () => {
        const decoder = SAM2_TINY_MODEL.files[1];
        const urls = getSam2ModelUrls(decoder, 'https://re-qi.github.io/ReSplat_Editor/');
        expect(urls.map(url => url.toString())).toEqual([
            'https://re-qi.github.io/ReSplat_Editor/static/models/sam2/sam2_hiera_tiny.decoder.onnx',
            decoder.remoteUrl
        ]);
    });

    it('keeps bundled models first for local and self-hosted builds', () => {
        const urls = getSam2ModelUrls(encoder, 'https://editor.example.com/app/');
        expect(urls.map(url => url.toString())).toEqual([
            'https://editor.example.com/app/static/models/sam2/sam2_hiera_tiny.encoder.onnx',
            encoder.remoteUrl
        ]);
    });
});
