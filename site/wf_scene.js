// Wrong Floor — the 3D layer: an elevator car that opens onto a fog town.
// PSX look: tiny render target upscaled with nearest filtering, vertices
// snapped to that grid, procedural canvas textures, exponential fog.
// The game (wf_game.js) drives it through setFloor / doors / tokens.
import * as THREE from "three";

const RES_W = 320, RES_H = 200;
const W = 2.2, H = 2.6, D = 2.2, DOOR_W = 1.2, DOOR_H = 2.1;

// ---------- procedural textures ----------
function canvasTex(w, h, draw) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.magFilter = THREE.NearestFilter; t.minFilter = THREE.NearestFilter;
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
  t.userData.canvas = c;
  return t;
}
function rnd(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
function noise(g, w, h, base, amp, cell, seed) {
  const r = rnd(seed);
  for (let y = 0; y < h; y += cell) for (let x = 0; x < w; x += cell) {
    const v = base + (r() - .5) * amp;
    g.fillStyle = `rgb(${v},${v + 3},${v + 2})`; g.fillRect(x, y, cell, cell);
  }
}
const TEX = {
  metal: () => canvasTex(64, 64, (g, w, h) => { noise(g, w, h, 150, 40, 4, 7); g.fillStyle = "rgba(255,255,255,.08)"; g.fillRect(0, 20, w, 10); }),
  door: () => canvasTex(32, 64, (g, w, h) => { noise(g, w, h, 140, 26, 2, 3); g.fillStyle = "rgba(0,0,0,.25)"; g.fillRect(w - 2, 0, 2, h); }),
  floor: () => canvasTex(64, 64, (g, w, h) => { noise(g, w, h, 60, 30, 8, 11); g.strokeStyle = "rgba(0,0,0,.4)"; for (let i = 0; i <= w; i += 16) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, h); g.moveTo(0, i); g.lineTo(w, i); g.stroke(); } }),
  asphalt: () => canvasTex(64, 64, (g, w, h) => noise(g, w, h, 55, 22, 2, 19)),
  walk: () => canvasTex(64, 64, (g, w, h) => { noise(g, w, h, 105, 26, 4, 23); g.strokeStyle = "rgba(0,0,0,.35)"; g.strokeRect(0, 0, w, h); }),
  facade: (seed, lit) => canvasTex(64, 128, (g, w, h) => {
    noise(g, w, h, 70, 30, 4, seed); const r = rnd(seed + 1);
    for (let y = 10; y < h - 20; y += 22) for (let x = 6; x < w - 8; x += 20) {
      const on = r() < lit; g.fillStyle = on ? "#c9a25a" : "#14120f"; g.fillRect(x, y, 10, 14);
    }
    g.fillStyle = "#0b0a09"; g.fillRect(20, h - 22, 22, 22);
  }),
  panel: () => canvasTex(32, 64, (g, w, h) => {
    noise(g, w, h, 165, 20, 2, 5); g.fillStyle = "#0a0f0a"; g.fillRect(5, 5, 22, 9);
    g.fillStyle = "#2b2b2b"; for (let i = 0; i < 4; i++) { g.beginPath(); g.arc(16, 24 + i * 9, 3, 0, 7); g.fill(); }
    g.fillStyle = "#9e1b16"; g.beginPath(); g.arc(16, 58, 3, 0, 7); g.fill();
  }),
  text: (w, h, bg, fg, lines, font) => canvasTex(w, h, (g) => {
    g.fillStyle = bg; g.fillRect(0, 0, w, h); g.fillStyle = fg; g.textAlign = "center"; g.textBaseline = "middle";
    lines.forEach((l, i) => { g.font = font[i] || font[0]; g.fillText(l, w / 2, h * (i + 1) / (lines.length + 1)); });
  }),
};

// ---------- PSX vertex snap ----------
function psx(mat) {
  mat.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace("#include <project_vertex>",
      `#include <project_vertex>
       vec2 g = vec2(${RES_W / 2}.0, ${RES_H / 2}.0);
       gl_Position.xy = floor(gl_Position.xy / gl_Position.w * g + .5) / g * gl_Position.w;`);
  };
  return mat;
}
const lambert = (o) => psx(new THREE.MeshLambertMaterial(o));

export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.setPixelRatio(1); renderer.setSize(RES_W, RES_H, false);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(68, RES_W / RES_H, 0.05, 80);
  camera.rotation.order = "YXZ";
  const eye = new THREE.Vector3(0, 1.6, 0.35); camera.position.copy(eye);
  scene.fog = new THREE.FogExp2(0x8a948f, 0.05); scene.background = new THREE.Color(0x8a948f);
  const hemi = new THREE.HemisphereLight(0xb8c4bf, 0x1a1612, 0.55); scene.add(hemi);

  // ---------- the car ----------
  const car = new THREE.Group(); scene.add(car);
  const metal = TEX.metal(); metal.repeat.set(2, 2);
  const wallM = lambert({ map: metal });
  const plane = (w, h, m) => new THREE.Mesh(new THREE.PlaneGeometry(w, h, 2, 2), m);
  const back = plane(W, H, wallM); back.position.set(0, H / 2, D / 2); back.rotation.y = Math.PI; car.add(back);
  const left = plane(D, H, wallM); left.position.set(-W / 2, H / 2, 0); left.rotation.y = Math.PI / 2; car.add(left);
  const right = plane(D, H, wallM); right.position.set(W / 2, H / 2, 0); right.rotation.y = -Math.PI / 2; car.add(right);
  const side = (W - DOOR_W) / 2;
  [-1, 1].forEach((s) => { const p = plane(side, H, wallM); p.position.set(s * (DOOR_W / 2 + side / 2), H / 2, -D / 2); car.add(p); });
  const lintel = plane(DOOR_W, H - DOOR_H, wallM); lintel.position.set(0, DOOR_H + (H - DOOR_H) / 2, -D / 2); car.add(lintel);
  const fl = TEX.floor(); fl.repeat.set(2, 2);
  const floor = plane(W, D, lambert({ map: fl })); floor.rotation.x = -Math.PI / 2; car.add(floor);
  const ceil = plane(W, D, lambert({ color: 0x9aa39f })); ceil.rotation.x = Math.PI / 2; ceil.position.y = H; car.add(ceil);
  const lamp = new THREE.Mesh(new THREE.CircleGeometry(0.22, 8), new THREE.MeshBasicMaterial({ color: 0xf1efe6 }));
  lamp.rotation.x = Math.PI / 2; lamp.position.set(0, H - 0.01, 0.2); car.add(lamp);
  const carLight = new THREE.PointLight(0xe9efe9, 2.2, 5, 1.4); carLight.position.set(0, H - 0.3, 0.2); car.add(carLight);
  const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, D * 0.8, 6), lambert({ color: 0x777f7b }));
  rail.rotation.x = Math.PI / 2; rail.position.set(-W / 2 + 0.07, 0.95, 0); car.add(rail);
  const rail2 = rail.clone(); rail2.position.x = W / 2 - 0.07; car.add(rail2);

  const doorM = lambert({ map: TEX.door() });
  const doors = [-1, 1].map((s) => {
    const d = new THREE.Mesh(new THREE.BoxGeometry(DOOR_W / 2, DOOR_H, 0.04), doorM);
    d.position.set(s * DOOR_W / 4, DOOR_H / 2, -D / 2 - 0.03); d.userData.s = s; car.add(d); return d;
  });
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(0.26, 0.52), lambert({ map: TEX.panel(), emissive: 0x111111 }));
  panel.position.set(W / 2 - 0.005, 1.25, -0.62); panel.rotation.y = -Math.PI / 2; panel.name = "panel"; car.add(panel);
  const indTex = TEX.text(64, 24, "#050805", "#d24a2a", ["1"], ["bold 18px monospace"]);
  const ind = plane(0.42, 0.16, new THREE.MeshBasicMaterial({ map: indTex }));
  ind.position.set(0, DOOR_H + 0.25, -D / 2 + 0.01); car.add(ind);

  // what stands behind you at the top
  const rider = makeFigure(1.75); rider.position.set(0.05, 0, D / 2 - 0.2); rider.rotation.y = Math.PI; rider.visible = false; car.add(rider);

  // ---------- the town ----------
  const town = new THREE.Group(); scene.add(town);
  const asp = TEX.asphalt(); asp.repeat.set(4, 20);
  const road = plane(8, 60, lambert({ map: asp })); road.rotation.x = -Math.PI / 2; road.position.set(0, -0.02, -31); town.add(road);
  const wk = TEX.walk(); wk.repeat.set(2, 30);
  [-1, 1].forEach((s) => { const p = plane(4, 60, lambert({ map: wk })); p.rotation.x = -Math.PI / 2; p.position.set(s * 6, 0.01, -31); town.add(p); });
  const facades = [];
  for (let i = 0; i < 6; i++) [-1, 1].forEach((s) => {
    const h = 7 + ((i * 7 + (s > 0 ? 3 : 0)) % 5) * 1.6, tex = TEX.facade(i * 2 + (s > 0), 0.5);
    tex.repeat.set(1, Math.round(h / 4));
    const b = new THREE.Mesh(new THREE.BoxGeometry(4, h, 9.6), lambert({ map: tex }));
    b.position.set(s * 10, h / 2, -6 - i * 10); town.add(b); facades.push(b);
  });
  const lampLights = [], lampHeads = [];
  for (let i = 0; i < 3; i++) {
    const s = i % 2 ? 1 : -1, z = -4 - i * 9;
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 4.2, 5), lambert({ color: 0x2a2826 }));
    post.position.set(s * 4.3, 2.1, z); town.add(post);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.18, 0.3), new THREE.MeshBasicMaterial({ color: 0xffd38a }));
    head.position.set(s * 4.05, 4.2, z); town.add(head); lampHeads.push(head);
    const L = new THREE.PointLight(0xffc77a, 6, 14, 1.6); L.position.set(s * 4.0, 3.9, z); town.add(L); lampLights.push(L);
  }
  // the poster wall: MISSING, the subject at the floor's dose
  const posterCanvas = document.createElement("canvas"); posterCanvas.width = 64; posterCanvas.height = 96;
  const posterTex = new THREE.CanvasTexture(posterCanvas); posterTex.magFilter = posterTex.minFilter = THREE.NearestFilter; posterTex.colorSpace = THREE.SRGBColorSpace;
  const board = new THREE.Mesh(new THREE.BoxGeometry(1.4, 2.0, 0.08), lambert({ color: 0x3a342c })); board.position.set(-2.1, 1.3, -5.5); board.rotation.y = 0.3; town.add(board);
  const poster = plane(0.9, 1.35, lambert({ map: posterTex })); poster.position.set(0, 0.05, 0.05); board.add(poster);
  // shop signs carry the lens tokens; the gallery window carries exp60
  const signs = [0, 1, 2].map((i) => {
    const s = i % 2 ? -1 : 1, m = new THREE.MeshBasicMaterial({ map: TEX.text(128, 24, "#120d0a", "#c9a227", [""], ["bold 15px monospace"]) });
    const p = plane(3.2, 0.6, m); p.position.set(s * 7.98, 3.4, -9 - i * 11); p.rotation.y = s > 0 ? -Math.PI / 2 : Math.PI / 2; town.add(p); return p;
  });
  const galCanvas = document.createElement("canvas"); galCanvas.width = 64; galCanvas.height = 64;
  const galTex = new THREE.CanvasTexture(galCanvas); galTex.magFilter = galTex.minFilter = THREE.NearestFilter; galTex.colorSpace = THREE.SRGBColorSpace;
  // an easel in the road holds the floor's exp60 painting
  const easel = new THREE.Group(); easel.position.set(1.9, 0, -7); easel.rotation.y = -0.35; town.add(easel);
  [-0.4, 0.4].forEach((x) => { const leg = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.9, 0.05), lambert({ color: 0x3a2a1c })); leg.position.set(x, 0.95, 0); leg.rotation.z = x * -0.12; easel.add(leg); });
  const gal = plane(1.1, 1.1, lambert({ map: galTex, emissive: 0x222222 })); gal.position.set(0, 1.45, 0.04); easel.add(gal);
  const walker = makeFigure(1.8); walker.position.set(0.6, 0, -30); town.add(walker);

  // ---------- state ----------
  const S = { yaw: 0, pitch: -0.05, door: 0, doorTarget: 0, travel: 0, shake: 0, tok: 0, flick: 0.04,
              base: { lamp: 6, car: 2.2 }, t: 0, dragging: false, flash: 0 };
  function setLook(dy, dp) { S.yaw += dy; S.pitch = Math.max(-1.0, Math.min(0.9, S.pitch + dp)); }

  function setFloor(f) {
    // f: { r (reading, units), void, poster(img|null), signs[], gallerySvgImg, figureDist, label }
    const r = Math.max(0, Math.min(8, f.r));
    const k = r / 8;
    const fog = new THREE.Color(0x8a948f).lerp(new THREE.Color(0x2b140e), Math.min(1, k * 1.15));
    if (f.void) fog.set(0x000000);
    scene.fog.color.copy(fog); scene.background = fog.clone();
    scene.fog.density = f.void ? 0.9 : 0.035 + k * 0.11;
    hemi.intensity = f.void ? 0.05 : 0.55 - k * 0.38;
    town.visible = !f.void;
    S.base.lamp = 7 - k * 5; S.flick = 0.04 + k * 0.5;
    walker.visible = !!f.figureDist; if (f.figureDist) walker.position.z = -f.figureDist;
    drawPoster(f.poster, f.label);
    signs.forEach((p, i) => { const w = (f.signs || [])[i] || ""; const t = TEX.text(128, 24, "#120d0a", "#c9a227", [w.toUpperCase()], ["bold 15px monospace"]); p.material.map.dispose(); p.material.map = t; });
    drawGallery(f.galleryImg, f.galleryCode);
    const it = indTex.userData.canvas.getContext("2d"); it.fillStyle = "#050805"; it.fillRect(0, 0, 64, 24);
    it.fillStyle = "#d24a2a"; it.font = "bold 18px monospace"; it.textAlign = "center"; it.textBaseline = "middle"; it.fillText(f.label, 32, 13); indTex.needsUpdate = true;
  }
  function drawPoster(img, label) {
    const g = posterCanvas.getContext("2d"); g.fillStyle = "#d8cbb4"; g.fillRect(0, 0, 64, 96);
    g.fillStyle = "#9e1b16"; g.font = "bold 11px monospace"; g.textAlign = "center"; g.fillText("MISSING", 32, 12);
    if (img && img.complete && img.naturalWidth) g.drawImage(img, 8, 17, 48, 56); else { g.fillStyle = "#222"; g.fillRect(8, 17, 48, 56); }
    g.fillStyle = "#222"; g.font = "7px monospace"; g.fillText("LAST SEEN", 32, 82); g.fillText("FLOOR " + label, 32, 91);
    posterTex.needsUpdate = true;
  }
  function drawGallery(img, code) {
    const g = galCanvas.getContext("2d"); g.fillStyle = "#050508"; g.fillRect(0, 0, 64, 64);
    if (code) { g.fillStyle = "#e04a3a"; g.font = "4px monospace"; code.split("\n").slice(0, 14).forEach((l, i) => g.fillText(l.trim().slice(0, 22), 3, 6 + i * 4.3)); }
    else if (img && img.complete && img.naturalWidth) g.drawImage(img, 2, 2, 60, 60);
    g.strokeStyle = "#c9a227"; g.lineWidth = 2; g.strokeRect(1, 1, 62, 62); galTex.needsUpdate = true;
  }

  const ray = new THREE.Raycaster();
  function pick(nx, ny) { ray.setFromCamera({ x: nx, y: ny }, camera); return ray.intersectObject(panel).length > 0; }
  function facing(target) { // 'panel' | 'back' | 'door'
    const y = ((S.yaw % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
    if (target === "back") return Math.abs(y) > 2.3;
    if (target === "panel") return y < -0.6 && y > -1.6;
    return Math.abs(y) < 0.6;
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now; S.t += dt;
    S.door += (S.doorTarget - S.door) * Math.min(1, dt * 1.6);
    doors.forEach((d) => { d.position.x = d.userData.s * (DOOR_W / 4 + S.door * DOOR_W / 2 * 0.98); });
    const tokenPulse = S.tok; // 0..~8 live projection of the token on screen
    lampLights.forEach((L, i) => {
      const f = 1 - S.flick * (0.5 + 0.5 * Math.sin(S.t * (7 + i * 3) + Math.sin(S.t * 23 + i) * 3)) - tokenPulse * 0.06;
      L.intensity = Math.max(0, S.base.lamp * f); lampHeads[i].material.color.setScalar(0.35 + 0.65 * Math.max(0, f));
    });
    carLight.intensity = S.base.car * (S.travel ? 0.85 + 0.15 * Math.sin(S.t * 40) : 1) * (1 - S.flash * 0.9);
    const sh = S.travel * 0.006 + S.shake;
    camera.position.set(eye.x + (Math.random() - .5) * sh, eye.y + (Math.random() - .5) * sh, eye.z);
    S.shake *= 0.9;
    camera.rotation.set(S.pitch, S.yaw, 0);
    if (walker.visible && S.door > 0.5) walker.rotation.y = Math.sin(S.t * 0.3) * 0.05;
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return {
    setFloor, pick, facing, setLook, S,
    openDoors() { S.doorTarget = 1; return wait(2600); },
    closeDoors() { S.doorTarget = 0; return wait(2600); },
    travel(on) { S.travel = on ? 1 : 0; },
    token(v) { S.tok = v; },
    jolt(v) { S.shake = v; },
    showRider(on) { rider.visible = on; },
    flash(v) { S.flash = v; },
    look(yaw, pitch) { S.yaw = yaw; S.pitch = pitch; },
  };
}

function makeFigure(h) {
  const g = new THREE.Group(), black = new THREE.MeshBasicMaterial({ color: 0x050404 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.24, h * 0.62, 6), black); body.position.y = h * 0.47; g.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 6, 5), black); head.position.y = h * 0.86; g.add(head);
  [-1, 1].forEach((s) => { const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, h * 0.36, 5), black); leg.position.set(s * 0.09, h * 0.18, 0); g.add(leg); });
  const eyeM = new THREE.MeshBasicMaterial({ color: 0xe8e2c8, fog: false });
  [-1, 1].forEach((s) => { const e = new THREE.Mesh(new THREE.PlaneGeometry(0.035, 0.012), eyeM); e.position.set(s * 0.045, h * 0.87, 0.125); g.add(e); });
  return g;
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
