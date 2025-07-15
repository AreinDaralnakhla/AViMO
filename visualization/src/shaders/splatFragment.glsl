
precision highp float;
precision highp int;

#include <splatDefines>

uniform bool encodeLinear;
uniform float maxStdDev;
uniform bool disableFalloff;
uniform float falloff;

out vec4 fragColor;

in vec4 vRgba;
in vec2 vSplatUv;
in vec3 vNdc;

void main() {
    float z = dot(vSplatUv, vSplatUv);
    if (z > (maxStdDev * maxStdDev)) {
        discard;
    }

    float alpha = vRgba.a;
    alpha *= mix(1.0, exp(-0.5 * z), falloff);
    if (alpha < MIN_ALPHA) {
        discard;
    }

    vec3 rgb = vRgba.rgb;
    if (encodeLinear) {
        rgb = srgbToLinear(rgb);
    }
    fragColor = vec4(rgb, alpha);
}
