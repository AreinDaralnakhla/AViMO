/**
 * @import { Shader } from '../../platform/graphics/shader.js'
 * @import { StencilParameters } from '../../platform/graphics/stencil-parameters.js'
 */
/**
 * A render pass that implements rendering a quad with a shader, and exposes controls over the
 * render state. This is typically used as a base class for render passes that render a quad with
 * a shader, but can be used directly as well by specifying a shader.
 *
 * @ignore
 */
export class RenderPassShaderQuad extends RenderPass {
    /**
     * @type {Shader|null}
     */
    _shader: Shader | null;
    /**
     * @type {QuadRender|null}
     */
    quadRender: QuadRender | null;
    /**
     * The cull mode to use when rendering the quad. Defaults to {@link CULLFACE_NONE}.
     */
    cullMode: number;
    /**
     * A blend state to use when rendering the quad. Defaults to {@link BlendState.NOBLEND}.
     *
     * @type {BlendState}
     */
    blendState: BlendState;
    /**
     * A depth state to use when rendering the quad. Defaults to {@link DepthState.NODEPTH}.
     *
     * @type {DepthState}
     */
    depthState: DepthState;
    /**
     * Stencil parameters for front faces to use when rendering the quad. Defaults to null.
     *
     * @type {StencilParameters|null}
     */
    stencilFront: StencilParameters | null;
    /**
     * Stencil parameters for back faces to use when rendering the quad. Defaults to null.
     *
     * @type {StencilParameters|null}
     */
    stencilBack: StencilParameters | null;
    /**
     * Sets the shader used to render the quad.
     *
     * @type {Shader}
     * @ignore
     */
    set shader(shader: Shader);
    get shader(): Shader;
}
import { RenderPass } from '../../platform/graphics/render-pass.js';
import type { Shader } from '../../platform/graphics/shader.js';
import { QuadRender } from './quad-render.js';
import { BlendState } from '../../platform/graphics/blend-state.js';
import { DepthState } from '../../platform/graphics/depth-state.js';
import type { StencilParameters } from '../../platform/graphics/stencil-parameters.js';
