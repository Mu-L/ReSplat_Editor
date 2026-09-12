import { mkdir, open, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const expectedFiles = [
    '洛带古镇.ply',
    'Bedroom.ply',
    'cat girl.ply',
    'Created by jimrgood.ply',
    'EVE.ply',
    'HIM.ply',
    'Le Mont Saint Michel, France.ply',
    'red and black nails.ply',
    'TeTo.ply',
    'YDTG.ply',
    'Zhuyufeng_point_cloud_1.ply',
    'ZhuYuFeng-ShouChi.ply',
    'ZYL_hq.ply'
];

const dataRoot = path.resolve(process.argv[2] ?? path.join(process.cwd(), '..', '数据'));
const reportRoot = path.join(process.cwd(), 'temp', 'segment-select-benchmark');

const readHeader = async (filename) => {
    const filePath = path.join(dataRoot, filename);
    const handle = await open(filePath, 'r');
    try {
        const stats = await handle.stat();
        const buffer = Buffer.alloc(Math.min(stats.size, 1024 * 1024));
        const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
        const source = buffer.subarray(0, bytesRead).toString('latin1');
        const end = source.indexOf('end_header');
        if (end === -1) throw new Error(`${filename}: PLY header exceeds 1 MiB or is missing end_header`);
        const header = source.slice(0, end + 'end_header'.length);
        if (!header.startsWith('ply')) throw new Error(`${filename}: invalid PLY magic`);
        const format = /^format\s+(\S+)/m.exec(header)?.[1];
        const vertices = Number(/^element\s+vertex\s+(\d+)/m.exec(header)?.[1]);
        if (!format || !Number.isSafeInteger(vertices) || vertices <= 0) {
            throw new Error(`${filename}: missing format or vertex count`);
        }
        return {
            filename,
            bytes: stats.size,
            sizeMiB: Number((stats.size / 1024 / 1024).toFixed(1)),
            format,
            vertices
        };
    } finally {
        await handle.close();
    }
};

const scenes = [];
for (const filename of expectedFiles) scenes.push(await readHeader(filename));

await mkdir(reportRoot, { recursive: true });
const reportPath = path.join(reportRoot, 'dataset-inventory.json');
await writeFile(reportPath, `${JSON.stringify({ generatedAt: new Date().toISOString(), dataRoot, scenes }, null, 2)}\n`);

console.table(scenes.map(({ filename, sizeMiB, format, vertices }) => ({ filename, sizeMiB, format, vertices })));
console.log(`Validated ${scenes.length} Segment Select reference PLY files.`);
console.log(`Report: ${reportPath}`);
