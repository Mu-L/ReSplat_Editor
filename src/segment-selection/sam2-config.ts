const SAM2_INPUT_SIZE = 1024;
const SAM2_MASK_INPUT_SIZE = 256;

type Sam2ModelFile = {
    label: 'encoder' | 'decoder';
    url: string;
    sha256: string;
    expectedBytes?: number;
};

type Sam2ModelConfig = {
    id: 'sam2-hiera-tiny';
    displayName: string;
    revision: string;
    files: readonly [Sam2ModelFile, Sam2ModelFile];
};

const REVISION = 'cd5a2fabd994d7b67e98d95df1232276b5cc8c0a';
const MODEL_ROOT = 'static/models/sam2';

const SAM2_TINY_MODEL: Sam2ModelConfig = {
    id: 'sam2-hiera-tiny',
    displayName: 'SAM2 Hiera Tiny',
    revision: REVISION,
    files: [
        {
            label: 'encoder',
            url: `${MODEL_ROOT}/sam2_hiera_tiny.encoder.onnx`,
            sha256: '4cc015ee18520e93f8c7ddfeaca7436039daaaaf19721b4b96a8810a805e82f7',
            expectedBytes: 134261315
        },
        {
            label: 'decoder',
            url: `${MODEL_ROOT}/sam2_hiera_tiny.decoder.onnx`,
            sha256: 'f5a4bd656c143899fb7f52d64ed81e6f6aeb37d477a0b6da50146ac7cf2187bf',
            expectedBytes: 20640886
        }
    ]
};

export {
    SAM2_INPUT_SIZE,
    SAM2_MASK_INPUT_SIZE,
    SAM2_TINY_MODEL
};
export type { Sam2ModelConfig, Sam2ModelFile };
