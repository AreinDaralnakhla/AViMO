export declare function unpackPcSogs(fileBytes: Uint8Array, extraFiles: Record<string, ArrayBuffer>): Promise<{
    packedArray: Uint32Array;
    numSplats: number;
    extra: Record<string, unknown>;
}>;
