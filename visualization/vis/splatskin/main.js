import GUI from "lil-gui";
import * as THREE from "three";

import {
  SparkRenderer,
  SparkControls,
  SplatMesh,
  constructAxes,
  constructGrid,
} from "@sparkjsdev/spark";

import {
  animateButterfly,
  makeBoneGrid,
} from "./bones.js";

const path = window.location.pathname;
const currentDir = path.substring(
  path.lastIndexOf("/", path.length - 2) + 1,
  path.length - 1,
);
const URL_BASE = "https://sparkjs.dev/assets/splats";

export function makeButterflyBones() {
  return new SplatMesh({
    constructSplats: async (splats) => {
      const origins = [];
      for (let y = -7; y <= 7; y++) {
        for (let x = -7; x <= 7; x++) {
          origins.push(new THREE.Vector3(x * 0.05, y * 0.05, 0));
        }
      }
      constructAxes({
        splats,
        origins,
        scale: 0.01,
        axisShadowScale: 1.5,
        axisRadius: 0.001,
      });
    },
    onLoad: makeBoneGrid,
  });
}

async function main() {
  const canvas = document.getElementById("canvas");
  const renderer = new THREE.WebGLRenderer({ canvas });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff); // Set background to white

  const forgeRenderer = new SparkRenderer({ renderer });
  const camera = new THREE.PerspectiveCamera(
    75,
    canvas.width / canvas.height,
    0.1,
    1000,
  );
  scene.add(forgeRenderer);
  scene.add(camera);

  const gui = new GUI({
    title: "Settings",
  });
  const guiOptions = {
    butterfly: true,
    skeleton: false,
  };
  gui.add(guiOptions, "butterfly").name("Show butterfly");
  gui.add(guiOptions, "skeleton").name("Show skeleton");
  
  const butterfly = new SplatMesh({
    url: `${URL_BASE}/butterfly.spz`,
    onLoad: makeBoneGrid,
  });
  
  butterfly.quaternion.set(1, 0, 0, 0);
  butterfly.position.set(0, 0, -1);
  butterfly.opacity = guiOptions.butterfly ? 1 : 0;
  scene.add(butterfly);

  const bones = makeButterflyBones();
  bones.quaternion.copy(butterfly.quaternion);
  bones.position.copy(butterfly.position);
  bones.opacity = guiOptions.skeleton ? 1 : 0;
  scene.add(bones);

  const grid = new SplatMesh({
    constructSplats: async (splats) =>
      constructGrid({
        splats,
        extents: new THREE.Box3(
          new THREE.Vector3(-10, -10, -10),
          new THREE.Vector3(10, 10, 10),
        ),
      }),
    onLoad: (mesh) => {
      mesh.editable = false;
    },
  });
  scene.add(grid);

  function handleResize() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  handleResize();
  window.addEventListener("resize", handleResize);

  const controls = new SparkControls({ canvas });

  console.log("Starting render loop");
  let lastTime;

  renderer.setAnimationLoop((time, _xrFrame) => {
    const timeSeconds = time * 0.001;
    if (lastTime == null) {
      lastTime = timeSeconds;
    }
    const deltaTime = timeSeconds - lastTime;
    lastTime = timeSeconds;
    // timeSeconds and deltaTime (from last frame) in seconds

    controls.update(camera);

    renderer.render(scene, camera);

    // Update state

    // butterfly.rotation.x = Math.PI + 0.5 * Math.sin(time);
    // butterfly.rotation.y += deltaTime;

    animateButterfly(butterfly, timeSeconds);
    animateButterfly(bones, timeSeconds);

    butterfly.opacity = Math.max(0, Math.min(1, butterfly.opacity + (guiOptions.butterfly ? 1 : -1) * deltaTime));
    bones.opacity = Math.max(0, Math.min(1, bones.opacity + (guiOptions.skeleton ? 1 : -1) * deltaTime));
    bones.visible = guiOptions.skeleton;
  });
}

main().catch(console.error);