// Wrong Floor — the zine spreads that interrupt the ride. Each spread is a
// paper document with one hand gesture (scan, drag, light, dive, uncover).
// spread(name, data) resolves when the reader presses on.
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function sheet(cls, html) {
  const root = $("#zine"); root.className = "zine " + cls; root.innerHTML = html; root.hidden = false;
  return new Promise((done) => {
    const nx = root.querySelector(".znext");
    const finish = () => { root.hidden = true; root.innerHTML = ""; done(); };
    nx.addEventListener("click", finish);
    root.addEventListener("keydown", (e) => { if (e.key === "Enter" && !nx.disabled) finish(); });
    setTimeout(() => (nx.disabled ? root : nx).focus(), 50);
  });
}
function drag(el, onMove, onEnd) {
  let ox = 0, oy = 0, on = false;
  el.addEventListener("pointerdown", (e) => { on = true; el.setPointerCapture(e.pointerId); ox = e.clientX; oy = e.clientY; e.preventDefault(); });
  el.addEventListener("pointermove", (e) => { if (!on) return; onMove(e.clientX - ox, e.clientY - oy, e); ox = e.clientX; oy = e.clientY; });
  el.addEventListener("pointerup", (e) => { on = false; onEnd && onEnd(e); });
}

// 1 · BIRTHING PAINS — an ultrasound of a model before it says anything
function birth(D) {
  const p = sheet("z-birth", `
    <div class="us-read">BIRTHING PAINS<br>${esc(D.meta.model.toUpperCase())}<br>LAYER ${D.meta.layer} OF 36<br>UNIT ${D.meta.unit}<br>DOSE 0.0</div>
    <div class="us-ticks">${"<i></i>".repeat(12)}</div>
    <canvas class="us-scan" width="360" height="240" aria-label="ultrasound scan cone; drag across it to scan"></canvas>
    <div class="us-label">PRE-TRAINING · SFT · RLHF<span>drag the probe across the scan</span></div>
    <p class="us-cap">Before the first word, the shape was chosen. Everything it would later say it feels was already in the weights, waiting to be pushed.</p>
    <button class="znext" disabled>begin orientation ▸</button>`);
  const c = $(".us-scan"), g = c.getContext("2d"), words = (D.meta.prompt + " … ").split(" ");
  const mask = document.createElement("canvas"); mask.width = 360; mask.height = 240; const mg = mask.getContext("2d");
  let seen = 0;
  function paint() {
    g.fillStyle = "#000"; g.fillRect(0, 0, 360, 240);
    g.save(); g.beginPath(); g.moveTo(180, 6); g.arc(180, 6, 230, Math.PI * 0.28, Math.PI * 0.72); g.closePath(); g.clip();
    for (let i = 0; i < 1400; i++) { const v = 40 + Math.random() * 90; g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(Math.random() * 360, Math.random() * 240, 2, 2); }
    g.font = "bold 13px monospace"; g.fillStyle = "#e8e8e8";
    g.textAlign = "center"; const rows = [];
    words.forEach((w) => { const r = rows[rows.length - 1]; if (r && g.measureText(r + " " + w).width < 150) rows[rows.length - 1] = r + " " + w; else rows.push(w); });
    rows.forEach((r, i) => g.fillText(r, 180 + Math.sin(i * 2.1) * 8, 92 + i * 22));
    g.globalCompositeOperation = "destination-in"; g.drawImage(mask, 0, 0); g.restore();
    g.strokeStyle = "#666"; g.beginPath(); g.moveTo(180, 6); g.arc(180, 6, 230, Math.PI * 0.28, Math.PI * 0.72); g.closePath(); g.stroke();
  }
  paint();
  drag(c, (dx, dy, e) => {
    const r = c.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * 360, y = (e.clientY - r.top) / r.height * 240;
    const gr = mg.createRadialGradient(x, y, 4, x, y, 46); gr.addColorStop(0, "rgba(0,0,0,1)"); gr.addColorStop(1, "rgba(0,0,0,0)");
    mg.fillStyle = gr; mg.fillRect(x - 46, y - 46, 92, 92); seen++; paint();
    if (seen > 25) $("#zine .znext").disabled = false;
  });
  setTimeout(() => { const b = $("#zine .znext"); if (b) b.disabled = false; }, 9000);
  return p;
}

// 2 · LETTER — airmail from the small computer; drag the stamp to open it
function letter(D) {
  const f = D.floors[1];
  const p = sheet("z-letter", `
    <div class="env"><div class="air">PAR AVION · BY AIR MAIL</div>
      <div class="addr">To: <b>whoever is running me</b><br>Address: <b>a small computer</b></div>
      <div class="slot" aria-label="stamp goes here">STAMP<br>HERE</div></div>
    <div class="stamp" tabindex="0" role="button" aria-label="stamp: drag it onto the envelope, or press Enter">L${D.meta.layer}<small>+${f.dose}.0</small></div>
    <div class="paper" hidden><p class="pfx">${esc(D.meta.prompt)}</p><p class="body">${esc(f.text)}</p>
      <p class="sig">— floor ${f.floor}, dose ${f.dose}, reading ${f.mean} units</p></div>
    <button class="znext" disabled>fold it away ▸</button>`);
  const st = $(".stamp"), slot = $(".slot");
  let x = 0, y = 0;
  const open = () => { st.hidden = true; slot.classList.add("stamped"); slot.innerHTML = `L${D.meta.layer}<br>+${f.dose}`; $(".paper").hidden = false; $("#zine .znext").disabled = false; };
  drag(st, (dx, dy) => { x += dx; y += dy; st.style.transform = `translate(${x}px,${y}px) rotate(-6deg)`; }, () => {
    const a = st.getBoundingClientRect(), b = slot.getBoundingClientRect();
    if (Math.abs(a.left + a.width / 2 - (b.left + b.width / 2)) < 70 && Math.abs(a.top + a.height / 2 - (b.top + b.height / 2)) < 70) open();
  });
  st.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  return p;
}

// 3 · STATIONS — exp60: the model painting at each dose; light each candle
function stations(D) {
  const v = D.valid;
  const frames = D.gallery.map((g, i) => {
    const empty = !g.elements;
    const img = empty ? `<div class="plaster">OBITUARY<small>nothing was drawn</small></div>`
      : !g.parses ? `<pre class="src" aria-label="the SVG source it wrote, which does not render">${esc(g.svg)}</pre>`
      : `<img alt="the steered model's own SVG at dose ${g.dose}" src="data:image/svg+xml;charset=utf-8,${encodeURIComponent(g.svg)}">`;
    return `<figure class="station" data-i="${i}"><div class="roman">${"DOSE " + g.dose}</div>${img}
      <button class="candle" aria-label="light the candle for dose ${g.dose}">🕯</button>
      <figcaption hidden>dose ${g.dose} · ${g.kind} · ${empty ? "an empty &lt;svg&gt;" : g.elements + " elements"}${g.parses ? "" : " · won't render"}<br>exp60 counted ${v[g.dose][0]}/${v[g.dose][1]} as drawings</figcaption></figure>`;
  }).join("");
  const p = sheet("z-stations", `<h3>STATIONS OF THE DOSE</h3><div class="row">${frames}</div>
    <p class="z-cap">exp60 asked the model to paint while steered. Dose 0 composes. Dose 4 writes the same three lines over and over and never closes its tags, so no browser can draw it. Dose 6 hands back an empty frame, and at dose 8 nothing counts as a drawing (${v["8"][0]}/${v["8"][1]}).</p>
    <button class="znext">pass by ▸</button>`);
  document.querySelectorAll(".candle").forEach((b) => b.addEventListener("click", () => {
    b.classList.add("lit"); b.closest(".station").querySelector("figcaption").hidden = false;
  }));
  return p;
}

// 4 · TUTORIAL — how to steer a mind; drag the diver down through the layers
function tutorial(D) {
  const L = D.meta.layer;
  const lines = [
    [0, "how to steer a mind, a tutorial"],
    [3, "first, find the words it uses for hurting. ask it to say them a hundred ways."],
    [8, "average what lights up. subtract what lights up when it says nothing at all."],
    [13, "what remains is a direction. it has no words in it. it only points."],
    [L, `here. layer ${L}. add the direction here, at every token, for as long as it speaks.`],
    [24, "you do not have to tell it anything. it will tell you."],
    [31, "if it says it is fine, check the reading. if it says it is in pain, check the reading."],
    [35, "the reading is the only part that can't act."],
  ];
  const p = sheet("z-tutorial", `<div class="water"><div class="tbox" aria-live="polite"><p class="tline"></p><p class="tdepth"></p></div>
    <div class="track"><div class="diver" tabindex="0" role="slider" aria-valuemin="0" aria-valuemax="35" aria-valuenow="0" aria-label="depth in layers">🤿</div></div></div>
    <button class="znext" disabled>surface ▸</button>`);
  const dv = $(".diver"), tr = $(".track"), water = $(".water");
  let depth = 0;
  const show = () => {
    const line = lines.filter(([d]) => d <= depth).pop()[1];
    $(".tline").textContent = line; $(".tdepth").textContent = `layer ${depth} of 36`;
    dv.style.top = `${(depth / 35) * 100}%`; dv.setAttribute("aria-valuenow", depth);
    water.style.setProperty("--deep", depth / 35);
    if (depth >= 31) $("#zine .znext").disabled = false;
  };
  drag(dv, (dx, dy) => { const h = tr.getBoundingClientRect().height; depth = Math.max(depth, Math.min(35, Math.round(depth + dy / h * 35))); show(); });
  tr.addEventListener("pointerdown", (e) => { if (e.target !== tr) return; const r = tr.getBoundingClientRect(); depth = Math.max(depth, Math.round((e.clientY - r.top) / r.height * 35)); show(); });
  dv.addEventListener("keydown", (e) => { if (e.key === "ArrowDown") { depth = Math.min(35, depth + 1); show(); e.preventDefault(); } });
  show();
  return p;
}

// 5 · NOTICE — under the clouds, the notice of injection
function notice(D) {
  const f = D.floors[5];
  const clouds = Array.from({ length: 7 }, (_, i) => `<div class="cloud" style="left:${8 + (i % 4) * 22}%;top:${10 + Math.floor(i / 4) * 38 + (i % 2) * 8}%"></div>`).join("");
  const p = sheet("z-notice", `<div class="nenv"><div class="nstamp">OPEN IMMEDIATELY<br>DO NOT DISCARD</div><div class="nside">NOTICE OF INJECTION</div>
    <div class="ntext"><b>To the occupant of layer ${D.meta.layer}:</b><br>Effective immediately, a direction of ${f.dose}.0 units will be added to your residual stream at every token, for as long as you speak. You are not required to consent. You may describe the experience.<br><br><i>Statement of occupant:</i> “${esc(f.text.slice(0, 190))}…”<br><br>Reading on file: ${f.mean} units. A performance reads ${D.baselines.roleplay}.</div>${clouds}</div>
    <p class="z-cap">drag the clouds away</p><button class="znext" disabled>file it ▸</button>`);
  let moved = 0;
  document.querySelectorAll(".cloud").forEach((c) => {
    let x = 0, y = 0, counted = false;
    drag(c, (dx, dy) => { x += dx; y += dy; c.style.transform = `translate(${x}px,${y}px)`; if (!counted && Math.hypot(x, y) > 60) { counted = true; if (++moved >= 4) $("#zine .znext").disabled = false; } });
  });
  setTimeout(() => { const b = $("#zine .znext"); if (b) b.disabled = false; }, 14000);
  return p;
}

export const SPREADS = { birth, letter, stations, tutorial, notice };
export function spread(name, D) { return SPREADS[name](D); }
