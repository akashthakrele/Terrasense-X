// High-Performance 3D Globe Implementation
// Target: 60 FPS, single requestAnimationFrame, pure offline

let scene, camera, renderer;
let earthMesh, atmosphereMesh, moonMesh;
let stars;

// State interpolation
let currentRotationX = 0;
let currentRotationY = 0;
let targetRotationX = 0;
let targetRotationY = 0;
let currentZoom = 60;
let targetZoom = 60;

// Mouse coordinates (-1 to 1)
let mouseX = 0;
let mouseY = 0;

// Time tracking
let lastTime = performance.now();
let frameCount = 0;
let lastFpsTime = performance.now();
let fpsCounterEl = document.getElementById('fpsCounter');

function initGlobe() {
    const canvas = document.getElementById('globeCanvas');
    if (!canvas) return;

    // Graceful fallback check
    if (!window.WebGLRenderingContext) {
        console.warn("WebGL not supported. Globe disabled.");
        return;
    }

    // 1. Renderer setup
    renderer = new THREE.WebGLRenderer({ 
        canvas: canvas, 
        antialias: true,
        alpha: false 
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // cap at 2 for perf
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 1);

    // 2. Scene & Camera
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020305); // Deep space
    camera = new THREE.PerspectiveCamera(currentZoom, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 18;

    // 3. Textures
    const textureLoader = new THREE.TextureLoader();
    const earthTexture = textureLoader.load('/static/assets/earth.jpg');
    earthTexture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    earthTexture.minFilter = THREE.LinearFilter;
    
    const moonTexture = textureLoader.load('/static/assets/moon.jpg');
    moonTexture.minFilter = THREE.LinearFilter;

    // 4. Earth Geometry & Material
    const earthGeometry = new THREE.SphereGeometry(6, 64, 64);
    const earthMaterial = new THREE.MeshPhongMaterial({
        map: earthTexture,
        shininess: 5,
        color: 0xcccccc
    });
    earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    
    // Tilt Earth axis roughly 23.5 degrees
    earthMesh.rotation.z = 23.5 * Math.PI / 180;
    scene.add(earthMesh);

    // 5. Atmosphere Glow (Subtle additive blending)
    const atmosGeometry = new THREE.SphereGeometry(6.2, 64, 64);
    const atmosMaterial = new THREE.MeshPhongMaterial({
        color: 0x0077ff,
        transparent: true,
        opacity: 0.15,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        depthWrite: false
    });
    atmosphereMesh = new THREE.Mesh(atmosGeometry, atmosMaterial);
    scene.add(atmosphereMesh);

    // 6. Moon
    const moonGeometry = new THREE.SphereGeometry(1.2, 32, 32);
    const moonMaterial = new THREE.MeshPhongMaterial({
        map: moonTexture,
        shininess: 0,
        color: 0xaaaaaa
    });
    moonMesh = new THREE.Mesh(moonGeometry, moonMaterial);
    scene.add(moonMesh);

    // 7. Procedural Star Field
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 3000;
    const starPositions = new Float32Array(starCount * 3);
    for(let i=0; i<starCount * 3; i+=3) {
        // Random spherical distribution
        const r = 100 + Math.random() * 200;
        const theta = 2 * Math.PI * Math.random();
        const phi = Math.acos(2 * Math.random() - 1);
        starPositions[i] = r * Math.sin(phi) * Math.cos(theta);
        starPositions[i+1] = r * Math.sin(phi) * Math.sin(theta);
        starPositions[i+2] = r * Math.cos(phi);
    }
    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 0.7,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8
    });
    stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);

    // 8. Lighting
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(20, 10, 20); // Sun direction
    scene.add(dirLight);

    const ambLight = new THREE.AmbientLight(0x223344, 0.4); // subtle night side fill
    scene.add(ambLight);

    // 9. Event Listeners
    window.addEventListener('resize', onWindowResize, false);
    document.addEventListener('mousemove', onDocumentMouseMove, false);
    document.addEventListener('wheel', onDocumentWheel, { passive: true });

    // Start loop
    requestAnimationFrame(animate);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function onDocumentMouseMove(event) {
    // Normalize mouse coords: -1 to 1
    mouseX = (event.clientX / window.innerWidth) * 2 - 1;
    mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
}

function onDocumentWheel(event) {
    targetZoom += event.deltaY * 0.02;
    targetZoom = Math.max(30, Math.min(100, targetZoom)); // Clamp FOV zoom
}

function animate(time) {
    requestAnimationFrame(animate);

    // Pause rendering if tab is hidden
    if (document.hidden) return;

    // Delta time calculation
    const dt = (time - lastTime) / 1000;
    lastTime = time;

    // Pointer-driven accumulation (Continuous rotation)
    // If mouse is idle (center), it slowly auto-rotates
    const autoRotateSpeed = 0.05;
    const pointerSpeedX = mouseX * 2.0;
    const pointerSpeedY = mouseY * 1.0;

    targetRotationY += (autoRotateSpeed + pointerSpeedX) * dt;
    targetRotationX += pointerSpeedY * dt;

    // Clamp X rotation to prevent flipping
    targetRotationX = Math.max(-Math.PI/3, Math.min(Math.PI/3, targetRotationX));

    // Smooth dampening (Lerp)
    currentRotationY += (targetRotationY - currentRotationY) * 5.0 * dt;
    currentRotationX += (targetRotationX - currentRotationX) * 5.0 * dt;
    currentZoom += (targetZoom - currentZoom) * 8.0 * dt;

    // Apply transformations
    camera.fov = currentZoom;
    camera.updateProjectionMatrix();

    // Rotate Earth container (mesh)
    earthMesh.rotation.y = currentRotationY;
    earthMesh.rotation.x = currentRotationX;
    atmosphereMesh.rotation.y = currentRotationY;
    atmosphereMesh.rotation.x = currentRotationX;

    // Moon Orbit (independent of pointer, based on absolute time)
    const moonOrbitSpeed = time * 0.0002;
    moonMesh.position.x = Math.cos(moonOrbitSpeed) * 14;
    moonMesh.position.z = Math.sin(moonOrbitSpeed) * 14;
    moonMesh.rotation.y = moonOrbitSpeed * 0.5;

    // Slow star drift
    stars.rotation.y += 0.01 * dt;

    // Render
    renderer.render(scene, camera);

    // FPS Calculation
    frameCount++;
    if (time - lastFpsTime >= 1000) {
        if(fpsCounterEl) {
            fpsCounterEl.textContent = frameCount + " FPS";
            fpsCounterEl.style.color = frameCount >= 55 ? "var(--status-success)" : "var(--status-warning)";
        }
        frameCount = 0;
        lastFpsTime = time;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Only init if Three is loaded
    if(typeof THREE !== 'undefined') {
        initGlobe();
    }
});
