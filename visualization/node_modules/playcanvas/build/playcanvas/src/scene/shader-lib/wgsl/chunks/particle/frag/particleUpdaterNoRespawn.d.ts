declare const _default: "\n    if (outLife >= uniform.lifetime) {\n        outLife = outLife - max(uniform.lifetime, (uniform.numParticles - 1.0) * particleRate);\n        visMode = -1.0;\n    }\n";
export default _default;
