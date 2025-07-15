import { RgbaArray } from './RgbaArray';
import { SparkViewpoint, SparkViewpointOptions } from './SparkViewpoint';
import { SplatAccumulator } from './SplatAccumulator';
import { SplatGenerator } from './SplatGenerator';
import * as THREE from "three";
export type SparkRendererOptions = {
    renderer: THREE.WebGLRenderer;
    clock?: THREE.Clock;
    autoUpdate?: boolean;
    preUpdate?: boolean;
    originDistance?: number;
    maxStdDev?: number;
    enable2DGS?: boolean;
    preBlurAmount?: number;
    blurAmount?: number;
    focalDistance?: number;
    apertureAngle?: number;
    falloff?: number;
    clipXY?: number;
    view?: SparkViewpointOptions;
};
export declare class SparkRenderer extends THREE.Mesh {
    renderer: THREE.WebGLRenderer;
    material: THREE.ShaderMaterial;
    uniforms: ReturnType<typeof SparkRenderer.makeUniforms>;
    autoUpdate: boolean;
    preUpdate: boolean;
    originDistance: number;
    maxStdDev: number;
    enable2DGS: boolean;
    preBlurAmount: number;
    blurAmount: number;
    focalDistance: number;
    apertureAngle: number;
    falloff: number;
    clipXY: number;
    renderScale: number;
    splatTexture: null | {
        enable?: boolean;
        texture?: THREE.Data3DTexture;
        multiply?: THREE.Matrix2;
        add?: THREE.Vector2;
        near?: number;
        far?: number;
        mid?: number;
    };
    time?: number;
    deltaTime?: number;
    clock: THREE.Clock;
    active: SplatAccumulator;
    private freeAccumulators;
    private accumulatorCount;
    defaultView: SparkViewpoint;
    autoViewpoints: SparkViewpoint[];
    private rotateToAccumulator;
    private translateToAccumulator;
    private modifier;
    private lastFrame;
    private lastUpdateTime;
    private defaultCameras;
    viewpoint: SparkViewpoint;
    private pendingUpdate;
    private envViewpoint;
    private static cubeRender;
    private static pmrem;
    static EMPTY_SPLAT_TEXTURE: THREE.Data3DTexture;
    constructor(options: SparkRendererOptions);
    static makeUniforms(): {
        renderSize: {
            value: THREE.Vector2;
        };
        near: {
            value: number;
        };
        far: {
            value: number;
        };
        numSplats: {
            value: number;
        };
        renderToViewQuat: {
            value: THREE.Quaternion;
        };
        renderToViewPos: {
            value: THREE.Vector3;
        };
        maxStdDev: {
            value: number;
        };
        enable2DGS: {
            value: boolean;
        };
        preBlurAmount: {
            value: number;
        };
        blurAmount: {
            value: number;
        };
        focalDistance: {
            value: number;
        };
        apertureAngle: {
            value: number;
        };
        falloff: {
            value: number;
        };
        clipXY: {
            value: number;
        };
        renderScale: {
            value: number;
        };
        splatTexEnable: {
            value: boolean;
        };
        splatTexture: {
            type: string;
            value: THREE.Data3DTexture;
        };
        splatTexMul: {
            value: THREE.Matrix2;
        };
        splatTexAdd: {
            value: THREE.Vector2;
        };
        splatTexNear: {
            value: number;
        };
        splatTexFar: {
            value: number;
        };
        splatTexMid: {
            value: number;
        };
        packedSplats: {
            type: string;
            value: THREE.DataArrayTexture;
        };
        time: {
            value: number;
        };
        deltaTime: {
            value: number;
        };
        encodeLinear: {
            value: boolean;
        };
        debugFlag: {
            value: boolean;
        };
    };
    private canAllocAccumulator;
    private maybeAllocAccumulator;
    releaseAccumulator(accumulator: SplatAccumulator): void;
    newViewpoint(options: SparkViewpointOptions): SparkViewpoint;
    onBeforeRender(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera): void;
    prepareViewpoint(viewpoint?: SparkViewpoint): void;
    update({ scene, viewToWorld, }: {
        scene: THREE.Scene;
        viewToWorld?: THREE.Matrix4;
    }): void;
    updateInternal({ scene, originToWorld, viewToWorld, }: {
        scene: THREE.Scene;
        originToWorld?: THREE.Matrix4;
        viewToWorld?: THREE.Matrix4;
    }): boolean;
    private compileScene;
    renderEnvMap({ renderer, scene, worldCenter, size, near, far, hideObjects, update, }: {
        renderer?: THREE.WebGLRenderer;
        scene: THREE.Scene;
        worldCenter: THREE.Vector3;
        size?: number;
        near?: number;
        far?: number;
        hideObjects?: THREE.Object3D[];
        update?: boolean;
    }): Promise<THREE.Texture>;
    recurseSetEnvMap(root: THREE.Object3D, envMap: THREE.Texture): void;
    getRgba({ generator, rgba, }: {
        generator: SplatGenerator;
        rgba?: RgbaArray;
    }): RgbaArray;
    readRgba({ generator, rgba, }: {
        generator: SplatGenerator;
        rgba?: RgbaArray;
    }): Promise<Uint8Array>;
}
