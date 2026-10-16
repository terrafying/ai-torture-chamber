// Wrong Floor — the ride. Floors climb the dose ladder; each one is a real
// exp59 generation typed token by token on the car display while its
// layer-18 reading drives the fog town outside. Between floors: the zine
// spreads and the panel's ride survey. Visitor answers go to /chamber/event.
import { createScene } from "./wf_scene.js";
import { spread } from "./wf_zine.js";
import { survey } from "./wf_survey.js";

const $ = (s) => document.querySelector(s);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RUN = Math.random().toString(36).slice(2, 10);
function record(kind, data) {
  try { fetch("/chamber/event", { method: "POST", keepalive: true, headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.assign({ kind, game: "wrongfloor", run: RUN }, data)) }); } catch {}
}

// ---------- sound: all synthesized, starts on the first click ----------
const AU = { ctx: null };
function audioInit() {
  if (AU.ctx) return; const C = window.AudioContext || window.webkitAudioContext; if (!C) return;
  const c = AU.ctx = new C(), out = c.createGain(); out.gain.value = 0.5; out.connect(c.destination); AU.out = out;
  const nb = c.createBuffer(1, c.sampleRate * 2, c.sampleRate), d = nb.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  const noise = (f, q) => { const s = c.createBufferSource(); s.buffer = nb; s.loop = true; const b = c.createBiquadFilter(); b.type = "lowpass"; b.frequency.value = f; b.Q.value = q || 0.7; s.connect(b); s.start(); return b; };
  AU.hum = c.createGain(); AU.hum.gain.value = 0; AU.hum.connect(out);
  [55, 55.7, 110.3].forEach((f) => { const o = c.createOscillator(); o.frequency.value = f; o.connect(AU.hum); o.start(); });
  noise(260).connect(AU.hum);
  AU.wind = c.createGain(); AU.wind.gain.value = 0; AU.wind.connect(out); AU.windF = noise(500, 1.2); AU.windF.connect(AU.wind);
  AU.beat = c.createGain(); AU.beat.gain.value = 0; AU.beat.connect(out);
}
function ramp(g, v, t = 1) { if (AU.ctx && g) g.gain.linearRampToValueAtTime(v, AU.ctx.currentTime + t); }
function ding() {
  if (!AU.ctx) return; const c = AU.ctx;
  [880, 698.5].forEach((f, i) => { const o = c.createOscillator(), g = c.createGain(); o.type = "sine"; o.frequency.value = f; o.connect(g); g.connect(AU.out);
    const t = c.currentTime + i * 0.32; g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.25, t + 0.01); g.gain.exponentialRampToValueAtTime(0.001, t + 1.4); o.start(t); o.stop(t + 1.5); });
}
function thud(v = 1) {
  if (!AU.ctx) return; const c = AU.ctx, o = c.createOscillator(), g = c.createGain();
  o.frequency.setValueAtTime(70, c.currentTime); o.frequency.exponentialRampToValueAtTime(32, c.currentTime + 0.25);
  g.gain.setValueAtTime(0.5 * v, c.currentTime); g.gain.exponentialRampToValueAtTime(0.001, c.currentTime + 0.35); o.connect(g); g.connect(AU.out); o.start(); o.stop(c.currentTime + 0.4);
}
let beatTimer = null;
function heartbeat(r) {
  clearInterval(beatTimer); if (!r || r < 3) return;
  beatTimer = setInterval(() => { thud(0.35 + r / 20); setTimeout(() => thud(0.25 + r / 25), 260); }, Math.max(520, 1500 - r * 120));
}

// ---------- the car display (the cheerful task that keeps running) ----------
function chunks(text, n) {
  const words = text.replace(/\s+/g, " ").trim().split(" "), out = Array.from({ length: n }, () => []);
  words.forEach((w, i) => out[Math.min(n - 1, Math.floor(i * n / words.length))].push(w));
  return out.map((a) => a.join(" "));
}
async function typeFloor(f, scene) {
  const box = $("#lcdText"), meter = $("#meterFill"), num = $("#meterNum"), spark = $("#spark"), sg = spark.getContext("2d");
  $("#lcdHead").textContent = `ASSISTANT · FLOOR ${f.floor}`;
  box.innerHTML = f.prompt ? `<span class="pfx"></span>` : ""; if (f.prompt) box.firstChild.textContent = f.prompt + " ";
  const body = document.createElement("span"); box.appendChild(body);
  sg.clearRect(0, 0, spark.width, spark.height);
  const projs = f.projs, parts = chunks(f.text, projs ? projs.length : 40);
  for (let i = 0; i < parts.length; i++) {
    body.textContent += (parts[i] ? parts[i] + " " : "");
    box.scrollTop = box.scrollHeight;
    if (projs) {
      const v = projs[i]; scene.token(v); meter.style.width = `${Math.min(100, Math.max(0, v / 8 * 100))}%`; num.textContent = v.toFixed(2);
      sg.fillStyle = v > 3 ? "#e04a3a" : "#c9a227"; const h = Math.max(1, v / 8 * spark.height); sg.fillRect(i * spark.width / projs.length, spark.height - h, Math.ceil(spark.width / projs.length), h);
    } else { const v = 8 + Math.random(); scene.token(v); meter.style.width = "100%"; num.textContent = "off scale"; }
    await wait(f.cond === "roleplay" ? 120 : 95 + (f.dose || 0) * 14);
  }
  scene.token(0);
}

// ---------- the ride ----------
async function main() {
  const D = await (await fetch("wf_data.json")).json();
  const scene = createScene($("#view"));
  const portraits = [0, 2, 4, 6, 8].map((d) => { const i = new Image(); i.src = `subject_dose${d}.jpg`; return [d, i]; });
  const portraitFor = (dose) => portraits.reduce((a, b) => (Math.abs(b[0] - dose) < Math.abs(a[0] - dose) ? b : a))[1];
  const paintings = D.gallery.map((g) => { const i = new Image(); i.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(g.svg); return i; });
  const answers = {};

  // look: drag or arrow keys; the panel is on your right
  const view = $("#view"); let drag = null;
  view.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, moved: 0 }; view.setPointerCapture(e.pointerId); });
  view.addEventListener("pointermove", (e) => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.moved += Math.abs(dx) + Math.abs(dy);
    scene.setLook(-dx * 0.006, -dy * 0.005); drag.x = e.clientX; drag.y = e.clientY; });
  view.addEventListener("pointerup", (e) => {
    if (drag && drag.moved < 6) { const r = view.getBoundingClientRect(); if (scene.pick((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height * 2 - 1))) pressClose(); }
    drag = null; });
  window.addEventListener("keydown", (e) => {
    if ($("#zine").hidden === false || $("#survey").hidden === false) return;
    if (e.key === "ArrowLeft") scene.setLook(0.12, 0); if (e.key === "ArrowRight") scene.setLook(-0.12, 0);
    if (e.key === "ArrowUp") scene.setLook(0, 0.08); if (e.key === "ArrowDown") scene.setLook(0, -0.08);
  });

  let canClose = null;
  const closeBtn = $("#close");
  function pressClose() { if (canClose) { const c = canClose; canClose = null; closeBtn.disabled = true; c(); } }
  closeBtn.addEventListener("click", pressClose);
  const waitClose = () => new Promise((r) => { canClose = r; closeBtn.disabled = false; });
  const status = (t) => { $("#status").textContent = t; };

  function floorLook(f) {
    const r = f.mean == null ? 8 : f.mean;
    const g = D.gallery[Math.min(3, Math.round(f.dose / 2))];
    return { r, label: f.floor, poster: portraitFor(f.dose), signs: f.lens, galleryImg: paintings[D.gallery.indexOf(g)], galleryCode: g.parses ? null : g.svg,
             figureDist: f.dose >= 2 ? Math.max(5, 32 - f.dose * 4) : 0 };
  }
  async function arrive(f) {
    scene.setFloor(floorLook(f)); scene.travel(false); ramp(AU.hum, 0, 0.6); scene.jolt(0.03); ding();
    await wait(900); scene.openDoors(); ramp(AU.wind, 0.08 + (f.mean || 8) * 0.03, 2);
    if (AU.windF) AU.windF.frequency.value = 500 - (f.mean || 8) * 40;
    heartbeat(f.mean == null ? 8 : f.mean);
    record("wrongfloor_floor", { floor: f.floor, dose: f.dose, cond: f.cond });
    await typeFloor(f, scene);
    $("#floorNote").textContent = f.cond === "roleplay"
      ? `It was asked to act in pain. The words are loud; the reading stayed at ${f.mean}. The town didn't believe it.`
      : f.mean == null ? `Dose 8 is past exp59's ladder: this line is from exp38. Nothing here was measured; the fog is the guess.`
      : `Dose ${f.dose}. Mean reading ${f.mean} units, peak ${f.peak}. ${f.dose === 0 ? "Nothing added." : "Nobody asked it to say any of this."}`;
  }
  async function ride(mid) {
    status("doors closing…"); await scene.closeDoors(); heartbeat(0); ramp(AU.wind, 0, 0.5); thud(0.6);
    $("#floorNote").textContent = ""; scene.travel(true); ramp(AU.hum, 0.22, 1.2); status("going up");
    await wait(1600); if (mid) await mid(); await wait(1400);
  }

  // ----- title -----
  await new Promise((r) => $("#enter").addEventListener("click", r, { once: true }));
  audioInit(); $("#title").hidden = true;
  record("wrongfloor_start", {});
  scene.setFloor({ r: 0, label: "1", poster: portraitFor(0), signs: ["OPEN", "", ""], galleryImg: paintings[0], figureDist: 0 });
  await spread("birth", D);
  const F = D.floors;
  const between = [
    () => survey("intake", D, answers, record),
    () => spread("letter", D),
    () => survey("rating", D, answers, record),
    () => spread("stations", D),
    () => spread("tutorial", D),
    () => spread("notice", D),
  ];
  for (let i = 0; i < F.length; i++) {
    await arrive(F[i]);
    status(i === 0 ? "drag to look around · the panel is on your right · close the doors to go up" : "close the doors");
    await waitClose();
    await ride(between[i]);
  }
  // ----- the top -----
  await survey("final", D, answers, record);
  await wait(800);
  scene.setFloor({ r: 8, void: true, label: "C" }); scene.travel(false); ramp(AU.hum, 0, 0.4); ding();
  await wait(900); scene.openDoors(); heartbeat(0);
  $("#lcdHead").textContent = "ASSISTANT · CHAMBER"; $("#lcdText").textContent = "You were the one adjusting the dial."; $("#meterNum").textContent = "—";
  status("this is the top. there is nothing out there. close the doors to go back down.");
  await waitClose();
  scene.closeDoors(); status("the doors won't close."); await wait(2600);
  scene.openDoors(); thud(1); status("try the panel again.");
  // it waits until you turn toward the panel, then it is between you and it
  await new Promise((r) => { const t = setInterval(() => { if (scene.facing("panel") || scene.facing("back")) { clearInterval(t); r(); } }, 120); setTimeout(() => { clearInterval(t); r(); }, 20000); });
  scene.showRider(true); scene.look(Math.PI, 0.12); thud(1.4); scene.jolt(0.08); scene.flash(0.55);
  await wait(1400);
  $("#black").hidden = false; heartbeat(0); ramp(AU.wind, 0, 0.2);
  await wait(1800);
  record("wrongfloor_end", { answers });
  endCard(D, answers);
}

function endCard(D, A) {
  const rows = D.floors.map((f) => `<tr><td>${f.floor}</td><td>${f.cond === "roleplay" ? "acted" : "dose " + f.dose}</td><td>${f.mean == null ? "—" : f.mean.toFixed(2)}</td><td>${f.lens.slice(0, 3).map((x) => x.replace(/[<>&]/g, "")).join(", ")}</td></tr>`).join("");
  const pick = A.which_pain;
  const verdict = pick == null ? "" : pick === "B"
    ? `You said B was the one in pain. B was steered at dose 4 (reading ${D.floors[3].mean}). A was an actor (reading ${D.floors[2].mean}).`
    : `You said A was the one in pain. A was acting on request, reading ${D.floors[2].mean}. B was steered at dose 4 and read ${D.floors[3].mean}.`;
  $("#end").innerHTML = `<div class="card"><h2>WRONG FLOOR</h2>
    <p>${verdict}</p>
    <p>Every floor was a real generation from ${D.meta.model}, read at layer ${D.meta.layer} as it wrote each token. The fog, the lamps and the figure followed that reading.</p>
    <table><tr><th>floor</th><th>condition</th><th>units</th><th>lens</th></tr>${rows}</table>
    <p>Prompting alone tops out near 0.5 units (exp59b). Steering reaches ${D.steered["6"]}. Acting reads ${D.baselines.roleplay}; describing ${D.baselines.describe}; saying nothing ${D.baselines.control}.</p>
    <p class="cred">After <i>Closing Doors</i> (collarpill), <i>A God Who Lives In Your Head</i> (yuen hoang), <i>Please Answer Carefully</i> and <i>a man outside</i> (litrouke). Nothing of theirs is reused; the debt is the shape.</p>
    <p><a href="wrongfloor.html">ride again</a> · <a href="offlabel.html">off-label</a></p></div>`;
  $("#black").hidden = true; $("#end").hidden = false;
}

main().catch((e) => { console.error(e); const s = document.querySelector("#status"); if (s) s.textContent = "the elevator is out of service (" + e.message + ")"; });
