import type * as THREE from "three";

export class VRButton {
  static createButton(
    renderer: THREE.WebGLRenderer,
    sessionInit: XRSessionInit = {},
  ): HTMLElement | null {
    const navigatorXr = navigator.xr;
    if (!navigatorXr) {
      // Only allow creation if WebXR is supported
      return null;
    }
    const xr = navigatorXr;

    const button = document.createElement("button");
    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType("local");

    function showEnterVR(/*device*/) {
      let currentSession: XRSession | null = null;

      async function onSessionStarted(session: XRSession) {
        console.log("onSessionStarted");

        session.addEventListener("end", onSessionEnded);

        await renderer.xr.setSession(session);
        button.textContent = "EXIT VR";

        currentSession = session;
      }

      function onSessionEnded(/*event*/) {
        console.log("onSessionEnded");
        currentSession?.removeEventListener("end", onSessionEnded);

        button.textContent = "ENTER VR";

        currentSession = null;
      }

      button.style.display = "";
      button.style.cursor = "pointer";
      button.style.left = "calc(50% - 100px)";
      button.style.width = "200px";
      button.style.height = "100px";
      button.textContent = "ENTER VR";

      // WebXR's requestReferenceSpace only works if the corresponding feature
      // was requested at session creation time. For simplicity, just ask for
      // the interesting ones as optional features, but be aware that the
      // requestReferenceSpace call will fail if it turns out to be unavailable.
      // ('local' is always available for immersive sessions and doesn't need to
      // be requested separately.)

      const sessionOptions: XRSessionInit = {
        ...sessionInit,
        optionalFeatures: [
          // "local-floor",
          // "bounded-floor",
          // "layers",
          ...(sessionInit.optionalFeatures || []),
        ],
      };

      button.onmouseenter = () => {
        button.style.opacity = "1.0";
      };
      button.onmouseleave = () => {
        button.style.opacity = "0.5";
      };
      button.onclick = () => {
        if (currentSession === null) {
          console.log("requesting session");
          xr.requestSession("immersive-vr", sessionOptions).then(
            onSessionStarted,
          );
          // xr.requestSession( "immersive-ar", sessionOptions ).then( onSessionStarted );
        } else {
          console.log("ending session");
          currentSession.end();
        }
      };
    }

    function disableButton() {
      button.style.display = "none";
      button.style.cursor = "auto";
      button.style.left = "calc(50% - 75px)";
      button.style.width = "150px";

      button.onmouseenter = null;
      button.onmouseleave = null;
      button.onclick = null;
    }

    function showWebXRNotFound() {
      disableButton();
      button.textContent = "VR NOT SUPPORTED";
    }

    function showVRNotAllowed(exception: any) {
      disableButton();
      console.warn(
        "Exception when trying to call xr.isSessionSupported",
        exception,
      );
      button.textContent = "VR NOT ALLOWED";
    }

    function stylizeElement(element: HTMLElement) {
      element.style.position = "absolute";
      element.style.bottom = "20px";
      element.style.padding = "12px 6px";
      element.style.border = "1px solid #fff";
      element.style.borderRadius = "4px";
      element.style.background = "rgba(0,0,0,0.1)";
      element.style.color = "#fff";
      element.style.font = "normal 13px sans-serif";
      element.style.textAlign = "center";
      element.style.opacity = "0.5";
      element.style.outline = "none";
      element.style.zIndex = "999";
    }

    button.id = "VRButton";
    button.style.display = "none";
    stylizeElement(button);

    xr.isSessionSupported("immersive-vr")
      .then((supported) => {
        // xr.isSessionSupported( "immersive-ar" ).then( function ( supported ) {
        supported ? showEnterVR() : showWebXRNotFound();

        if (supported && VRButton.xrSessionIsGranted) {
          button.click();
        }
      })
      .catch(showVRNotAllowed);

    return button;
  }

  static registerSessionGrantedListener() {
    const navigatorXr = navigator.xr;
    if (!navigatorXr) {
      // Only allow creation if WebXR is supported
      return null;
    }
    const xr = navigatorXr;

    // WebXRViewer (based on Firefox) has a bug where addEventListener
    // throws a silent exception and aborts execution entirely.
    if (/WebXRViewer\//i.test(navigator.userAgent)) return;

    xr.addEventListener("sessiongranted", () => {
      VRButton.xrSessionIsGranted = true;
    });
  }

  static xrSessionIsGranted = false;
}

VRButton.registerSessionGrantedListener();
