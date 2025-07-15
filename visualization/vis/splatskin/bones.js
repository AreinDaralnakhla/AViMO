import { SplatMesh, SplatSkinning } from "@sparkjsdev/spark";
import * as THREE from "three";

export function gridToBoneIndex(x, y) {
  return 15 * (y + 7) + (x + 7);
}

export async function makeBoneGrid(mesh) {
  const skinning = new SplatSkinning({ mesh, numBones: 15 * 15 });
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  for (let y = -7; y <= 7; y++) {
    for (let x = -7; x <= 7; x++) {
      position.set(x * 0.05, y * 0.05, 0);
      skinning.setRestQuatPos(gridToBoneIndex(x, y), quaternion, position);
      skinning.setBoneQuatPos(gridToBoneIndex(x, y), quaternion, position);
    }
  }

  const boneIndices = new THREE.Vector4();
  const weights = new THREE.Vector4();
  mesh.forEachSplat((index, center, scales, quaternion, opacity, color) => {
    const x = center.y / 0.05;
    const y = center.x / 0.05;
    const gridX = Math.max(-7, Math.min(6, Math.floor(x)));
    const gridY = Math.max(-7, Math.min(6, Math.floor(y)));
    const fractX = Math.max(0, Math.min(1, x - gridX));
    const fractY = Math.max(0, Math.min(1, y - gridY));

    boneIndices.set(
      gridToBoneIndex(gridX, gridY),
      gridToBoneIndex(gridX + 1, gridY),
      gridToBoneIndex(gridX, gridY + 1),
      gridToBoneIndex(gridX + 1, gridY + 1),
    );
    weights.set(
      (1 - fractX) * (1 - fractY),
      fractX * (1 - fractY),
      (1 - fractX) * fractY,
      fractX * fractY,
    );

    skinning.setSplatBones(index, boneIndices, weights);
  });
  mesh.skinning = skinning;
  mesh.updateGenerator();
}

export function animateBoneGrid(skinning, time) {
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  for (let y = -7; y <= 7; y++) {
    for (let x = -7; x <= 7; x++) {
      const px = x * 0.05;
      const py = y * 0.05;
      // position.set(px, py, 0);
      position.set(
        px + 0.1 * Math.sin(x * 0.5 + time) * 0.1,
        py + 0.1 * Math.sin(y * 0.5 + 1.3 * time),
        0.1 * Math.sin(time * 1.1 + px + py),
      );
      quaternion.setFromEuler(
        new THREE.Euler(
          x * 0.09 * Math.sin(time),
          y * 0.05 * Math.cos(time * 3.7),
          0.1 * Math.sin(time * 1.1),
        ),
      );
      skinning.setBoneQuatPos(gridToBoneIndex(x, y), quaternion, position);
    }
  }
  skinning.updateBones();
}

export function animateToad(toad, time) {
  const skinning = toad.skinning;
  if (!skinning) {
    return;
  }
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const SPEED = 5;
  for (let y = -7; y <= 7; y++) {
    for (let x = -7; x <= 7; x++) {
      const px = x * 0.05;
      const py = y * 0.05;
      position.set(px, py, 0);
      quaternion.setFromEuler(
        new THREE.Euler(
          0,
          y * -0.1 * Math.cos(time * SPEED),
          -0.2 * Math.cos(time * SPEED),
        ),
      );
      skinning.setBoneQuatPos(gridToBoneIndex(x, y), quaternion, position);
    }
  }
  skinning.updateBones();
}

export function animateButterfly(butterfly, time) {
  const skinning = butterfly.skinning;
  if (!skinning) {
    return;
  }
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  for (let y = -7; y <= 7; y++) {
    for (let x = -7; x <= 7; x++) {
      const px = x * 0.05;
      const py = y * 0.05;
      position.set(px, py, 0);
      quaternion.setFromEuler(
        new THREE.Euler(
          x * 0.09 * Math.sin(time * 1.1),
          y * 0.05 * Math.cos(time * 7.1),
          0.2 * Math.sin(time * 1.1),
        ),
      );
      skinning.setBoneQuatPos(gridToBoneIndex(x, y), quaternion, position);
    }
  }
  skinning.updateBones();
}