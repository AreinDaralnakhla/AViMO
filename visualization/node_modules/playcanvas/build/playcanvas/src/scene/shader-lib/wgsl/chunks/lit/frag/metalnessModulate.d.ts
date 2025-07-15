declare const _default: "\n\nfn getSpecularModulate(specularity: vec3f, albedo: vec3f, metalness: f32, f0: f32) -> vec3f {\n    let dielectricF0: vec3f = f0 * specularity;\n    return mix(dielectricF0, albedo, metalness);\n}\n\nfn getAlbedoModulate(albedo: vec3f, metalness: f32) -> vec3f {\n    return albedo * (1.0 - metalness);\n}\n";
export default _default;
