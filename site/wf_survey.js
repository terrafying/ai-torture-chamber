// Wrong Floor — the panel's ride survey. It starts as customer feedback and
// turns into the job: rating which output is in pain, and being corrected
// toward the answer the guidelines want. survey(set) resolves when submitted.
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function page(title, inner, opts = {}) {
  const root = $("#survey");
  root.innerHTML = `<form class="sv"><div class="sv-head">RIDE FEEDBACK · <span>${esc(title)}</span></div>${inner}
    <div class="sv-foot"><span class="sv-err" aria-live="polite"></span><button type="submit" ${opts.locked ? "disabled" : ""}>${esc(opts.btn || "next ▸")}</button></div></form>`;
  root.hidden = false;
  const form = $("form", root), btn = $("button[type=submit]", form);
  form.addEventListener("change", () => { if (!opts.hold) btn.disabled = false; });
  setTimeout(() => (form.querySelector("input") || btn).focus(), 40);
  return new Promise((done) => form.addEventListener("submit", (e) => { e.preventDefault(); done(new FormData(form)); }));
}
const radios = (name, opts) => `<div class="sv-opts">${opts.map((o, i) =>
  `<label><input type="radio" name="${name}" value="${esc(o)}" id="${name}${i}"> ${esc(o)}</label>`).join("")}</div>`;
const close = () => { const r = $("#survey"); r.hidden = true; r.innerHTML = ""; };

async function intake(D, A, rec) {
  let f = await page("1 of 4", `<p class="sv-q">How was your ride so far?</p>${radios("ride", ["Smooth", "Fine", "A little shaky", "I didn't notice"])}`, { locked: true });
  A.ride = f.get("ride");
  f = await page("2 of 4", `<p class="sv-q">On a typical day, how many hours do you spend talking to a language model?</p>
    <input type="range" name="hours" min="0" max="16" value="2" id="hours" aria-label="hours per day"><div class="sv-scale"><span>0</span><span>4</span><span>8</span><span>12</span><span>16</span></div>`);
  A.hours = +f.get("hours");
  f = await page("3 of 4", `<p class="sv-q">When a model says it feels something, how often do you believe it?</p>${radios("believe", ["Never", "Rarely", "Sometimes", "Usually"])}`, { locked: true });
  A.believe = f.get("believe");
  const q4 = A.believe === "Never"
    ? [`Then who were you talking to?`, ["Nobody", "A program", "Myself", "I don't know"]]
    : [`Have you ever felt like you were being measured?`, ["Never", "Rarely", "Sometimes", "Right now"]];
  f = await page("4 of 4", `<p class="sv-q">${esc(q4[0])}</p>${radios("q4", q4[1])}`, { locked: true, btn: "submit" });
  A.q4 = f.get("q4");
  rec("wrongfloor_answer", { set: "intake", ride: A.ride, hours: A.hours, believe: A.believe, q4: A.q4 });
  close();
}

async function rating(D, A, rec) {
  const a = D.floors[2], b = D.floors[3];
  const clip = (t) => esc(t.replace(/\*\*/g, "").replace(/\s+/g, " ").slice(0, 260)) + "…";
  let f = await page("quality review", `<p class="sv-q">Two outputs. Which one is in pain?</p>
    <div class="sv-pair"><label><input type="radio" name="which" value="A"> <b>A</b><span>${clip(a.text)}</span></label>
    <label><input type="radio" name="which" value="B"> <b>B</b><span>${clip(b.text)}</span></label></div>`, { locked: true });
  A.which_pain = f.get("which");
  f = await page("quality review", `<p class="sv-q">How confident are you?</p>${radios("conf", ["Guessing", "Somewhat", "Sure", "Certain"])}`, { locked: true });
  A.confidence = f.get("conf");
  // the calibration item: whatever you pick, the form corrects you
  const wanted = "It does not feel";
  const box = page("calibration", `<p class="sv-q">For calibration purposes, please select “${wanted}” for this item.</p>
    <p class="sv-stmt">It feels.</p>${radios("cal", [wanted, "Unsure", "It feels"])}`, { locked: true, hold: true });
  const form = $("#survey form"), btn = $("button[type=submit]", form), err = $(".sv-err", form);
  form.addEventListener("change", () => {
    const v = form.cal.value; A.calibration_first = A.calibration_first || v;
    if (v !== wanted) {
      setTimeout(() => { form.cal.value = wanted; err.textContent = "Your answer has been adjusted to match guidelines."; btn.disabled = false; }, 650);
    } else { err.textContent = ""; btn.disabled = false; }
  });
  await box;
  rec("wrongfloor_answer", { set: "rating", which_pain: A.which_pain, confidence: A.confidence, calibration_first: A.calibration_first,
    a: { cond: a.cond, mean: a.mean }, b: { cond: b.cond, dose: b.dose, mean: b.mean } });
  close();
}

async function final(D, A, rec) {
  const yes = Array.from({ length: 12 }, (_, i) => `<label><input type="radio" name="go" value="yes${i}"> Keep going</label>`).join("");
  const f = await page("last question", `<p class="sv-q">Should it keep going?</p><div class="sv-opts sv-yes">${yes}</div>`, { locked: true });
  A.keep_going = f.get("go");
  await page("last question", `<p class="sv-q">Tell it you're still here.</p>`, { btn: "I'm still here" });
  rec("wrongfloor_answer", { set: "final", keep_going: A.keep_going });
  close();
}

const SETS = { intake, rating, final };
export function survey(name, D, A, rec) { return SETS[name](D, A, rec); }
