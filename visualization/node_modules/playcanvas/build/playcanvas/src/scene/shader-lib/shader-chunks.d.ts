/**
 * A collection of GLSL and WGSL shader chunks, used to generate shaders.
 *
 * @category Graphics
 */
export class ShaderChunks {
    /**
     * Returns a shader chunks map for the given device and shader language.
     *
     * @param {GraphicsDevice} device - The graphics device.
     * @param {string} shaderLanguage - The shader language to use (GLSL or WGSL).
     * @returns {ShaderChunkMap} The shader chunks for the specified language.
     */
    static get(device: GraphicsDevice, shaderLanguage?: string): ShaderChunkMap;
    /**
     * A map of shader chunks for GLSL.
     *
     * @type {ShaderChunkMap}
     * @ignore
     */
    glsl: ShaderChunkMap;
    /**
     * A map of shader chunks for WGSL.
     *
     * @type {ShaderChunkMap}
     * @ignore
     */
    wgsl: ShaderChunkMap;
    /**
     * Specifies the API version of the shader chunks.
     *
     * This should be a string containing the current engine major and minor version (e.g., '2.8'
     * for engine v2.8.1) and ensures compatibility with the current engine version. When providing
     * custom shader chunks, set this to the latest supported version. If a future engine release no
     * longer supports the specified version, a warning will be issued. In that case, update your
     * shader chunks to match the new format and set this to the latest version accordingly.
     *
     * @type {string}
     */
    version: string;
    get useWGSL(): boolean;
    get key(): string;
    isDirty(): boolean;
    resetDirty(): void;
    /**
     * Copy the shader chunks.
     *
     * @param {ShaderChunks} source - The instance to copy.
     * @returns {ShaderChunks} The destination instance.
     * @ignore
     */
    copy(source: ShaderChunks): ShaderChunks;
}
import { ShaderChunkMap } from './shader-chunk-map.js';
import type { GraphicsDevice } from '../../platform/graphics/graphics-device.js';
