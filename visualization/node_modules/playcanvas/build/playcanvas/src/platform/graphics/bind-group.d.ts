/**
 * A bind group represents a collection of {@link UniformBuffer}, {@link Texture} and
 * {@link StorageBuffer} instanced, which can be bind on a GPU for rendering.
 *
 * @ignore
 */
export class BindGroup {
    /**
     * Create a new Bind Group.
     *
     * @param {GraphicsDevice} graphicsDevice - The graphics device used to manage this uniform buffer.
     * @param {BindGroupFormat} format - Format of the bind group.
     * @param {UniformBuffer} [defaultUniformBuffer] - The default uniform buffer. Typically a bind
     * group only has a single uniform buffer, and this allows easier access.
     */
    constructor(graphicsDevice: GraphicsDevice, format: BindGroupFormat, defaultUniformBuffer?: UniformBuffer);
    /**
     * A render version the bind group was last updated on.
     *
     * @type {number}
     * @private
     */
    private renderVersionUpdated;
    /** @type {UniformBuffer[]} */
    uniformBuffers: UniformBuffer[];
    /**
     * An array of offsets for each uniform buffer in the bind group. This is the offset in the
     * buffer where the uniform buffer data starts.
     *
     * @type {number[]}
     */
    uniformBufferOffsets: number[];
    id: number;
    device: GraphicsDevice;
    format: BindGroupFormat;
    dirty: boolean;
    impl: any;
    textures: any[];
    storageTextures: any[];
    storageBuffers: any[];
    /** @type {UniformBuffer} */
    defaultUniformBuffer: UniformBuffer;
    /**
     * Frees resources associated with this bind group.
     */
    destroy(): void;
    /**
     * Assign a uniform buffer to a slot.
     *
     * @param {string} name - The name of the uniform buffer slot
     * @param {UniformBuffer} uniformBuffer - The Uniform buffer to assign to the slot.
     */
    setUniformBuffer(name: string, uniformBuffer: UniformBuffer): void;
    /**
     * Assign a storage buffer to a slot.
     *
     * @param {string} name - The name of the storage buffer slot.
     * @param {StorageBuffer} storageBuffer - The storage buffer to assign to the slot.
     */
    setStorageBuffer(name: string, storageBuffer: StorageBuffer): void;
    /**
     * Assign a texture to a named slot.
     *
     * @param {string} name - The name of the texture slot.
     * @param {Texture} texture - Texture to assign to the slot.
     */
    setTexture(name: string, texture: Texture): void;
    /**
     * Assign a storage texture to a named slot.
     *
     * @param {string} name - The name of the texture slot.
     * @param {Texture} texture - Texture to assign to the slot.
     */
    setStorageTexture(name: string, texture: Texture): void;
    /**
     * Updates the uniform buffers in this bind group.
     */
    updateUniformBuffers(): void;
    /**
     * Applies any changes made to the bind group's properties. Note that the content of used
     * uniform buffers needs to be updated before calling this method.
     */
    update(): void;
}
/**
 * Data structure to hold a bind group and its offsets. This is used by {@link UniformBuffer#update}
 * to return a dynamic bind group and offset for the uniform buffer.
 *
 * @ignore
 */
export class DynamicBindGroup {
    bindGroup: any;
    offsets: any[];
}
import type { UniformBuffer } from './uniform-buffer.js';
import type { GraphicsDevice } from './graphics-device.js';
import type { BindGroupFormat } from './bind-group-format.js';
import type { StorageBuffer } from './storage-buffer.js';
import type { Texture } from './texture.js';
