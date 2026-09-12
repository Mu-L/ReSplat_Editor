// @ts-expect-error Test-only Node builtin; browser builds do not import this module.
import { createHash } from 'node:crypto';
// @ts-expect-error Test-only Node builtin; browser builds do not import this module.
import { createReadStream, statSync } from 'node:fs';
// @ts-expect-error Test-only Node builtin; browser builds do not import this module.
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { SAM2_TINY_MODEL } from './sam2-config';

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

            const filePath = fileURLToPath(new URL(`../../${file.url}`, import.meta.url));
            expect(statSync(filePath).size).toBe(file.expectedBytes);
            await expect(sha256File(filePath)).resolves.toBe(file.sha256);
        });
    }
});
