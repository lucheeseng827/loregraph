/* loregraph — PoC canvas renderer (vanilla JS, no build step, air-gap clean).
 *
 * This is the OSS PoC renderer: a small Canvas2D force simulation over a neighbor-first
 * graph. Per PLAN.md §6 the MVP renderer is vendored Cytoscape.js (Canvas, ~3-5k nodes)
 * and the scale tier is Sigma.js + graphology (WebGL). They sit behind the same
 * renderer-blind JSON contract this file already consumes (/v1/graph, /v1/graph/neighbors,
 * /v1/search, /v1/node/{id}), so swapping the renderer never touches the backend. */
"use strict";

// Keys are the backend's stable NodeKind tags (snake_case), matching /v1/* wire output.
const KIND_COLORS = {
  session: "#7aa2f7", turn: "#9aa5b1", decision: "#f7768e", implementation: "#9ece6a",
  pattern: "#e0af68", debt_signal: "#db4b4b", repo: "#bb9af7", module: "#7dcfff",
  file: "#73daca", symbol: "#2ac3de", manifest: "#cfc9c2", package: "#b4f9f8",
  commit: "#ff9e64", framework: "#c0caf5", tooling: "#a9b1d6", data_source: "#f7c8e0",
  design_doc: "#e6db74", concept: "#89ddff", person: "#ff75a0",
};
const color = (k) => KIND_COLORS[k] || "#8b949e";
const esc = (s) => { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; };

const cv = document.getElementById("canvas");
const ctx = cv.getContext("2d");
const state = {
  nodes: new Map(), edges: [], pos: new Map(), vel: new Map(),
  cam: { x: 0, y: 0, z: 1 }, hot: null, drag: null, panning: false,
  typesOff: new Set(),
};

function banner(msg) {
  const b = document.getElementById("banner");
  document.getElementById("banner-msg").textContent = msg;
  b.hidden = false;
}
document.getElementById("banner-x").onclick = () => (document.getElementById("banner").hidden = true);

async function api(path) {
  const r = await fetch(path, { headers: { accept: "application/json" } });
  if (!r.ok) throw new Error(path + " → " + r.status);
  return r.json();
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", () => { resize(); });

function seed(id) {
  if (state.pos.has(id)) return;
  const a = (Number(BigInt.asUintN(32, BigInt(id || 0))) % 360) * Math.PI / 180;
  const r = 80 + (id ? Number(BigInt.asUintN(16, BigInt(id))) % 240 : 0);
  state.pos.set(id, { x: Math.cos(a) * r, y: Math.sin(a) * r });
  state.vel.set(id, { x: 0, y: 0 });
}

function setGraph(g, merge) {
  // On a full replace, drop layout state too — otherwise pos/vel for gone nodes leak.
  if (!merge) { state.nodes.clear(); state.edges = []; state.pos.clear(); state.vel.clear(); }
  for (const n of g.nodes || []) { state.nodes.set(n.id, n); seed(n.id); }
  const seen = new Set(state.edges.map((e) => e.from + ">" + e.kind + ">" + e.to));
  for (const e of g.edges || []) { const k = e.from + ">" + e.kind + ">" + e.to; if (!seen.has(k)) { state.edges.push(e); seen.add(k); } }
  document.getElementById("stats").textContent = `${state.nodes.size} nodes · ${state.edges.length} edges`;
  renderFilters();
}

function renderFilters() {
  const kinds = [...new Set([...state.nodes.values()].map((n) => n.kind))].sort();
  const wrap = document.getElementById("filters");
  wrap.innerHTML = "";
  for (const k of kinds) {
    const c = document.createElement("span");
    c.className = "chip" + (state.typesOff.has(k) ? "" : " on");
    c.textContent = k;
    c.style.borderColor = state.typesOff.has(k) ? "" : color(k);
    c.onclick = () => { state.typesOff.has(k) ? state.typesOff.delete(k) : state.typesOff.add(k); renderFilters(); };
    wrap.appendChild(c);
  }
}

/* --- force simulation (Fruchterman-Reingold-ish, capped) --- */
function step() {
  const ids = [...state.nodes.keys()].filter((id) => !state.typesOff.has(state.nodes.get(id).kind));
  const K = 90;
  for (let i = 0; i < ids.length; i++) {
    const pi = state.pos.get(ids[i]); const vi = state.vel.get(ids[i]);
    let fx = -pi.x * 0.002, fy = -pi.y * 0.002; // gravity to center
    for (let j = 0; j < ids.length; j++) {
      if (i === j) continue;
      const pj = state.pos.get(ids[j]);
      let dx = pi.x - pj.x, dy = pi.y - pj.y; let d2 = dx * dx + dy * dy + 0.01;
      const rep = (K * K) / d2;
      const d = Math.sqrt(d2); fx += (dx / d) * rep; fy += (dy / d) * rep;
    }
    vi.fx = fx; vi.fy = fy;
  }
  for (const e of state.edges) {
    const a = state.pos.get(e.from), b = state.pos.get(e.to);
    if (!a || !b) continue;
    const va = state.vel.get(e.from), vb = state.vel.get(e.to);
    let dx = b.x - a.x, dy = b.y - a.y; const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
    const spring = (d - K) * 0.02;
    const ux = dx / d, uy = dy / d;
    if (va) { va.fx = (va.fx || 0) + ux * spring; va.fy = (va.fy || 0) + uy * spring; }
    if (vb) { vb.fx = (vb.fx || 0) - ux * spring; vb.fy = (vb.fy || 0) - uy * spring; }
  }
  for (const id of ids) {
    if (state.drag === id) continue;
    const p = state.pos.get(id), v = state.vel.get(id);
    v.x = (v.x + (v.fx || 0)) * 0.82; v.y = (v.y + (v.fy || 0)) * 0.82;
    const cap = 8; v.x = Math.max(-cap, Math.min(cap, v.x)); v.y = Math.max(-cap, Math.min(cap, v.y));
    p.x += v.x; p.y += v.y;
  }
}

function toScreen(p) {
  return { x: cv.clientWidth / 2 + (p.x + state.cam.x) * state.cam.z, y: cv.clientHeight / 2 + (p.y + state.cam.y) * state.cam.z };
}
function fromScreen(sx, sy) {
  return { x: (sx - cv.clientWidth / 2) / state.cam.z - state.cam.x, y: (sy - cv.clientHeight / 2) / state.cam.z - state.cam.y };
}

function draw() {
  step();
  ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(139,148,158,.28)";
  for (const e of state.edges) {
    const a = state.nodes.get(e.from), b = state.nodes.get(e.to);
    if (!a || !b || state.typesOff.has(a.kind) || state.typesOff.has(b.kind)) continue;
    const pa = toScreen(state.pos.get(e.from)), pb = toScreen(state.pos.get(e.to));
    ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
  }
  for (const n of state.nodes.values()) {
    if (state.typesOff.has(n.kind)) continue;
    const p = toScreen(state.pos.get(n.id));
    const r = (n.kind === "Decision" || n.kind === "Repo" || n.kind === "Session" ? 8 : 5) * Math.max(0.6, state.cam.z);
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = color(n.kind); ctx.fill();
    if (state.hot === n.id) { ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke(); }
    if (state.cam.z > 0.85) {
      ctx.fillStyle = "rgba(230,237,243,.85)"; ctx.font = "11px ui-sans-serif, sans-serif";
      ctx.fillText((n.label || "").slice(0, 28), p.x + r + 3, p.y + 3);
    }
  }
  requestAnimationFrame(draw);
}

function pick(sx, sy) {
  let best = null, bd = 14 * 14;
  for (const n of state.nodes.values()) {
    if (state.typesOff.has(n.kind)) continue;
    const p = toScreen(state.pos.get(n.id));
    const d = (p.x - sx) ** 2 + (p.y - sy) ** 2;
    if (d < bd) { bd = d; best = n.id; }
  }
  return best;
}

/* --- interaction --- */
let downPt = null, movedSinceDown = false;
cv.addEventListener("mousedown", (ev) => {
  downPt = { x: ev.offsetX, y: ev.offsetY };
  movedSinceDown = false;
  const id = pick(ev.offsetX, ev.offsetY);
  if (id) { state.drag = id; } else { state.panning = true; }
  state.last = { x: ev.offsetX, y: ev.offsetY };
});
window.addEventListener("mousemove", (ev) => {
  const rect = cv.getBoundingClientRect();
  const ox = ev.clientX - rect.left, oy = ev.clientY - rect.top;
  if (state.last) {
    const dx = ox - state.last.x, dy = oy - state.last.y;
    if (downPt && Math.abs(ox - downPt.x) + Math.abs(oy - downPt.y) > 4) movedSinceDown = true;
    if (state.drag) { const w = fromScreen(ox, oy); state.pos.set(state.drag, { x: w.x, y: w.y }); }
    else if (state.panning) { state.cam.x += dx / state.cam.z; state.cam.y += dy / state.cam.z; }
    state.last = { x: ox, y: oy };
  } else {
    // hover (no button held) — highlight the node under the cursor
    state.hot = pick(ox, oy);
  }
});
window.addEventListener("mouseup", () => { state.drag = null; state.panning = false; state.last = null; });
// Single open path: a click that wasn't a drag opens the node (no duplicate /v1 fetches).
cv.addEventListener("click", (ev) => {
  if (movedSinceDown) return;
  const id = pick(ev.offsetX, ev.offsetY);
  if (id) openNode(id);
});
cv.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const f = ev.deltaY < 0 ? 1.1 : 1 / 1.1;
  state.cam.z = Math.max(0.15, Math.min(4, state.cam.z * f));
}, { passive: false });

async function openNode(id) {
  try {
    const d = await api(`/v1/node/${encodeURIComponent(id)}`);
    const n = d.node || state.nodes.get(id);
    state.hot = id;
    document.getElementById("panel").hidden = false;
    document.getElementById("panel-title").textContent = n.label || "(node)";
    document.getElementById("panel-kind").textContent = n.kind;
    document.getElementById("panel-kind").style.color = color(n.kind);
    document.getElementById("panel-body").textContent = n.body || n.summary || "(no body)";
    const pv = n.provenance;
    document.getElementById("panel-prov").innerHTML = pv
      ? `<b>source</b> ${esc(pv.source_tool)}<br><b>session</b> ${esc(pv.native_session_id || "—")}<br>` +
        `<b>commit</b> ${esc(pv.commit || "—")}<br><b>extractor</b> ${esc(pv.extractor)} · conf ${esc((pv.confidence || 0).toFixed(2))}`
      : "<i>no provenance (repo-derived or synthetic)</i>";
    // expand neighbors on click — this is the anti-hairball "neighbor-first" call
    const sub = await api(`/v1/graph/neighbors?id=${encodeURIComponent(id)}&depth=1`);
    setGraph(sub, true);
    const ul = document.getElementById("panel-neighbors"); ul.innerHTML = "";
    for (const nb of (sub.nodes || []).filter((x) => x.id !== id).slice(0, 40)) {
      const li = document.createElement("li");
      li.innerHTML = `<span style="color:${color(nb.kind)}">●</span> ${esc(nb.label)}`;
      li.onclick = () => openNode(nb.id);
      ul.appendChild(li);
    }
  } catch (e) { banner("node load failed: " + e.message); }
}
document.getElementById("panel-x").onclick = () => (document.getElementById("panel").hidden = true);

async function doSearch() {
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  try {
    const d = await api(`/v1/search?q=${encodeURIComponent(q)}&k=12`);
    if (d.degraded) banner("no vector index loaded — showing lexical matches (build with --features localmodel for semantic recall)");
    if (!d.results || !d.results.length) { banner("nothing recalled for: " + q); return; }
    // pull each hit's neighborhood so the canvas focuses the recalled subgraph
    state.nodes.clear(); state.edges = []; state.pos.clear(); state.vel.clear();
    // allSettled (not all): one failed neighbor fetch must not drop the whole batch —
    // the graph was already cleared above, so render whatever neighborhoods succeed.
    const settled = await Promise.allSettled(
      d.results.map((hit) => api(`/v1/graph/neighbors?id=${encodeURIComponent(hit.id)}&depth=1`))
    );
    for (const r of settled) if (r.status === "fulfilled") setGraph(r.value, true);
    await openNode(d.results[0].id);
  } catch (e) { banner("search failed: " + e.message); }
}
document.getElementById("q-btn").onclick = doSearch;
document.getElementById("q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
document.getElementById("reload").onclick = boot;

async function boot() {
  resize();
  try { const g = await api("/v1/graph?limit=400"); setGraph(g, false); }
  catch (e) { banner("could not load graph: " + e.message + " — run `lore index` first, then `lore serve`."); }
}
boot();
draw();
