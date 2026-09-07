"""Baut einen lokalen HTML-Viewer fuer die Segmentierungsergebnisse.

Statische JPGs sind zum Beurteilen unpraktisch: man kann nicht zoomen, nicht
zwischen Original und Overlay hin- und herblenden und muss zum Vergleichen
Dateien in verschiedenen Fenstern oeffnen.

Der Viewer ist eine einzelne HTML-Datei, die die vorhandenen Bilder per relativem
Pfad einbindet -- nichts wird kopiert oder hochgeladen. Im Browser oeffnen:

    firefox results_views/viewer.html

Mit mehreren --variants laesst sich zwischen Verfahren umschalten, ohne dass sich
Zoom oder Bildausschnitt aendern -- so sieht man direkt, welches Verfahren eine
Krone anders schneidet.

Bedienung:
    1-5         Ebene wechseln (Original, Instanzen, Luecken, Relief, Trennungen)
    Q W E ...   Variante wechseln (bzw. Tab zum Durchschalten)
    Leertaste   Original einblenden solange gedrueckt (A/B-Vergleich)
    Pfeiltasten Frame wechseln
    Mausrad     Zoom, Ziehen verschiebt
    S           Split-Modus: Trennlinie zwischen Original und Ebene ziehen
    0           Ansicht zuruecksetzen

Beispiel:
    python make_viewer.py \
        --variants Hybrid=results_views SAM3=results_views_sam3 Multiskala=results_views_multiskala \
        --out vergleich.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infer_species import REPO_ROOT

VIEWS = [
    ("original", "Original"),
    ("instanzen", "Instanzen"),
    ("luecken", "Lücken"),
    ("relief", "Relief"),
    ("trennungen", "Trennungen"),
]

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Kronenabgrenzung — Viewer</title>
<style>
  :root { --bg:#12141a; --panel:#1c2029; --line:#2b3240; --text:#e6e9ef; --muted:#95a0b4; --accent:#5aa9ff; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
               font:14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  #app { display:grid; grid-template-columns: 250px 1fr; height:100%; }
  #side { background:var(--panel); border-right:1px solid var(--line); overflow-y:auto; padding:12px; }
  #side h1 { font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
             margin:0 0 10px; font-weight:600; }
  .folder { margin-bottom:10px; }
  .folder > div { font-size:12px; color:var(--muted); margin:8px 0 4px; }
  .frame { padding:4px 8px; border-radius:5px; cursor:pointer; font-size:13px; color:var(--text); }
  .frame:hover { background:#252b36; }
  .frame.active { background:var(--accent); color:#06101d; font-weight:600; }
  #main { display:flex; flex-direction:column; min-width:0; }
  #bar { display:flex; align-items:center; gap:8px; padding:10px 14px; border-bottom:1px solid var(--line);
         background:var(--panel); flex-wrap:wrap; }
  button { background:#252b36; color:var(--text); border:1px solid var(--line); border-radius:6px;
           padding:6px 11px; cursor:pointer; font-size:13px; }
  button.active { background:var(--accent); color:#06101d; border-color:var(--accent); font-weight:600; }
  #bar label { color:var(--muted); font-size:12px; display:flex; align-items:center; gap:6px; }
  input[type=range] { width:130px; accent-color: var(--accent); }
  #stage { flex:1; position:relative; overflow:hidden; background:#0b0d12; cursor:grab; }
  #stage.drag { cursor:grabbing; }
  #wrap { position:absolute; transform-origin:0 0; }
  #wrap img { position:absolute; top:0; left:0; display:block; max-width:none; user-select:none;
              -webkit-user-drag:none; }
  #caption { padding:6px 14px; border-top:1px solid var(--line); background:var(--panel);
             color:var(--muted); font-size:12px; display:flex; justify-content:space-between; }
  kbd { background:#252b36; border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:11px; }
  #divider { position:absolute; top:0; bottom:0; width:2px; background:var(--accent); display:none;
             pointer-events:none; }
</style>
<div id="app">
  <div id="side"><h1>Frames</h1><div id="frames"></div></div>
  <div id="main">
    <div id="bar">
      <span id="views"></span>
      <span id="variants" style="display:flex; gap:6px; border-left:1px solid var(--line); padding-left:8px;"></span>
      <label>Deckkraft <input type="range" id="alpha" min="0" max="100" value="100"></label>
      <button id="split">Split (S)</button>
      <button id="reset">Reset (0)</button>
    </div>
    <div id="stage"><div id="wrap"></div><div id="divider"></div></div>
    <div id="caption">
      <span id="info"></span>
      <span><kbd>1</kbd>–<kbd>5</kbd> Ebene · <kbd>Leer</kbd> Original · <kbd>←</kbd><kbd>→</kbd> Frame ·
            <kbd>Q</kbd><kbd>W</kbd><kbd>E</kbd>/<kbd>Tab</kbd> Variante · Rad = Zoom</span>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const VIEWS = __VIEWS__;
const VARIANTS = __VARIANTS__;
const VKEYS = ['q', 'w', 'e', 'r', 't'];

const stage = document.getElementById('stage');
const wrap = document.getElementById('wrap');
const divider = document.getElementById('divider');
const alphaInput = document.getElementById('alpha');

let frameIndex = 0, view = 1, variant = 0, splitMode = false, splitX = 0.5;
let scale = 1, tx = 0, ty = 0, fitScale = 1, showOriginal = false;
const imgs = {};

function buildSidebar() {
  const host = document.getElementById('frames');
  let current = null, group;
  DATA.forEach((f, i) => {
    if (f.folder !== current) {
      current = f.folder;
      group = document.createElement('div');
      group.className = 'folder';
      group.innerHTML = `<div>${f.folder}</div>`;
      host.appendChild(group);
    }
    const el = document.createElement('div');
    el.className = 'frame';
    el.textContent = f.name;
    el.onclick = () => selectFrame(i);
    el.dataset.index = i;
    group.appendChild(el);
  });
}

function buildViewButtons() {
  const host = document.getElementById('views');
  VIEWS.forEach((v, i) => {
    const b = document.createElement('button');
    b.textContent = `${i + 1} ${v[1]}`;
    b.onclick = () => setView(i);
    b.dataset.view = i;
    host.appendChild(b);
  });
}

function buildVariantButtons() {
  const host = document.getElementById('variants');
  if (VARIANTS.length < 2) return;
  VARIANTS.forEach((name, i) => {
    const b = document.createElement('button');
    b.textContent = `${(VKEYS[i] || '').toUpperCase()} ${name}`;
    b.onclick = () => setVariant(i);
    b.dataset.variant = i;
    host.appendChild(b);
  });
}

function loadFrame() {
  // Zoom und Ausschnitt bleiben erhalten -- nur so ist ein Variantenvergleich
  // ueberhaupt aussagekraeftig.
  const f = DATA[frameIndex];
  const name = VARIANTS[variant];
  const views = f.variants[name] || {};
  wrap.innerHTML = '';
  VIEWS.forEach((v, k) => {
    const img = document.createElement('img');
    img.src = k === 0 ? f.original : (views[v[0]] || f.original);
    img.style.zIndex = k;
    wrap.appendChild(img);
    imgs[k] = img;
  });
  const missing = !f.variants[name];
  document.getElementById('info').textContent =
    `${f.folder}/${f.name} — ${name}${missing ? ' (keine Ansicht)' : ''}`;
  document.querySelectorAll('#variants button').forEach(b =>
    b.classList.toggle('active', +b.dataset.variant === variant));
}

function selectFrame(i) {
  frameIndex = (i + DATA.length) % DATA.length;
  loadFrame();
  document.querySelectorAll('.frame').forEach(el =>
    el.classList.toggle('active', +el.dataset.index === frameIndex));
  imgs[0].onload = () => { fit(); render(); };
  render();
}

function setVariant(i) {
  variant = (i + VARIANTS.length) % VARIANTS.length;
  loadFrame();
  render();
}

function setView(i) {
  view = i;
  document.querySelectorAll('#views button').forEach(b =>
    b.classList.toggle('active', +b.dataset.view === view));
  render();
}

function fit() {
  const w = imgs[0].naturalWidth, h = imgs[0].naturalHeight;
  if (!w) return;
  fitScale = Math.min(stage.clientWidth / w, stage.clientHeight / h);
  scale = fitScale;
  tx = (stage.clientWidth - w * scale) / 2;
  ty = (stage.clientHeight - h * scale) / 2;
}

function render() {
  wrap.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  const alpha = alphaInput.value / 100;
  const active = showOriginal ? 0 : view;
  Object.entries(imgs).forEach(([k, img]) => {
    const isActive = +k === active;
    img.style.opacity = +k === 0 ? 1 : (isActive ? alpha : 0);
    img.style.display = (+k === 0 || isActive) ? 'block' : 'none';
    img.style.clipPath = (splitMode && isActive && +k !== 0)
      ? `inset(0 0 0 ${splitX * 100}%)` : 'none';
  });
  divider.style.display = splitMode ? 'block' : 'none';
  if (splitMode) {
    const w = imgs[0].naturalWidth * scale;
    divider.style.left = `${tx + splitX * w}px`;
  }
}

stage.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0015);
  const r = stage.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const next = Math.min(20 * fitScale, Math.max(0.2 * fitScale, scale * factor));
  tx = mx - (mx - tx) * (next / scale);
  ty = my - (my - ty) * (next / scale);
  scale = next;
  render();
}, { passive: false });

let dragging = false, lastX = 0, lastY = 0;
stage.addEventListener('mousedown', e => {
  if (splitMode && e.shiftKey) return;
  dragging = true; lastX = e.clientX; lastY = e.clientY; stage.classList.add('drag');
});
addEventListener('mousemove', e => {
  if (splitMode && !dragging) {
    const w = imgs[0].naturalWidth * scale;
    splitX = Math.min(1, Math.max(0, (e.clientX - stage.getBoundingClientRect().left - tx) / w));
    render();
    return;
  }
  if (!dragging) return;
  tx += e.clientX - lastX; ty += e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  render();
});
addEventListener('mouseup', () => { dragging = false; stage.classList.remove('drag'); });

addEventListener('keydown', e => {
  if (e.code === 'Space') { showOriginal = true; render(); e.preventDefault(); return; }
  if (e.key === 'Tab') { setVariant(variant + 1); e.preventDefault(); return; }
  const vk = VKEYS.indexOf(e.key.toLowerCase());
  if (vk >= 0 && vk < VARIANTS.length) { setVariant(vk); return; }
  if (e.key >= '1' && e.key <= String(VIEWS.length)) setView(+e.key - 1);
  else if (e.key === 'ArrowRight') selectFrame(frameIndex + 1);
  else if (e.key === 'ArrowLeft') selectFrame(frameIndex - 1);
  else if (e.key.toLowerCase() === 's') { splitMode = !splitMode; render(); }
  else if (e.key === '0') { fit(); render(); }
});
addEventListener('keyup', e => { if (e.code === 'Space') { showOriginal = false; render(); } });

alphaInput.oninput = render;
document.getElementById('split').onclick = () => { splitMode = !splitMode; render(); };
document.getElementById('reset').onclick = () => { fit(); render(); };
addEventListener('resize', () => { fit(); render(); });

buildSidebar();
buildViewButtons();
buildVariantButtons();
setView(1);
selectFrame(0);
</script>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views-dir", type=Path, default=REPO_ROOT / "results_views")
    parser.add_argument("--variants", nargs="*", default=None, metavar="NAME=VERZEICHNIS",
                        help="Mehrere Ansichtsverzeichnisse zum Umschalten, z.B. Hybrid=results_views "
                             "SAM3=results_views_sam3. Ohne Angabe wird --views-dir benutzt.")
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=None, help="Default: <views-dir>/viewer.html")
    return parser.parse_args()


def parse_variants(args) -> list[tuple[str, Path]]:
    if not args.variants:
        return [(args.views_dir.name, args.views_dir)]
    variants = []
    for item in args.variants:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--variants erwartet NAME=VERZEICHNIS, bekam: {item!r}")
        variants.append((name, Path(path)))
    return variants


def main() -> None:
    args = parse_args()
    variants = parse_variants(args)
    out_path = args.out or variants[0][1] / "viewer.html"
    base = out_path.parent.resolve()

    # Frames ueber alle Varianten einsammeln, damit auch Frames auftauchen, die
    # nur eine Variante segmentiert hat.
    frames: dict[tuple[str, str], dict] = {}
    for variant_name, views_dir in variants:
        if not views_dir.exists():
            print(f"  Variante {variant_name}: {views_dir} fehlt, uebersprungen")
            continue
        for view_folder in sorted(p for p in views_dir.iterdir() if p.is_dir()):
            stems = sorted({p.name.rsplit("_", 1)[0] for p in view_folder.glob("*_*.jpg")})
            for stem in stems:
                originals = list((args.input / view_folder.name).glob(f"{stem}.*"))
                if not originals:
                    continue
                entry = frames.setdefault(
                    (view_folder.name, stem),
                    {
                        "folder": view_folder.name,
                        "name": stem,
                        "original": originals[0].resolve().as_uri(),
                        "variants": {},
                    },
                )
                views = {}
                for key, _ in VIEWS[1:]:
                    candidate = (view_folder / f"{stem}_{key}.jpg").resolve()
                    if candidate.exists():
                        # Relativ zum Viewer, solange moeglich -- sonst absolut.
                        try:
                            views[key] = str(candidate.relative_to(base))
                        except ValueError:
                            views[key] = candidate.as_uri()
                if views:
                    entry["variants"][variant_name] = views

    ordered = [frames[k] for k in sorted(frames)]
    if not ordered:
        print("Keine Ansichten gefunden.")
        return

    names = [name for name, _ in variants]
    html = (
        TEMPLATE.replace("__DATA__", json.dumps(ordered))
        .replace("__VIEWS__", json.dumps(VIEWS))
        .replace("__VARIANTS__", json.dumps(names))
    )
    out_path.write_text(html, encoding="utf-8")

    print(f"{len(ordered)} Frames, Varianten: {', '.join(names)}")
    print(f"-> {out_path}")
    print(f"   firefox {out_path}")


if __name__ == "__main__":
    main()
