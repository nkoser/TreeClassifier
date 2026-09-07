"""Write an interactive 3D viewer as a single HTML file.

The result is a file you open in a browser -- no server, no installation, nothing
to fetch. Points, colours and the ground image are embedded in it, and the
drawing is done with WebGL.

The coordinates are stored as int16 rather than float32. That halves the file
size and costs nothing: for a scene a good 100 m across, the step size is about
3 mm, far below anything the depth estimate resolves.

Controls in the viewer: drag rotates, the wheel zooms, the right mouse button
pans. Colour, point size and a height threshold are adjustable.

    python depthft/viewer3d.py --frames 80m/frame_000297.jpg
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402
from punktwolke import bodenmodell, kamera_pruefen, kantenmaske  # noqa: E402

BILDENDUNGEN = {".jpg", ".jpeg", ".png"}

VORLAGE = """<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<title>__TITEL__</title>
<style>
  html,body{margin:0;height:100%;background:#111;color:#ddd;
            font:13px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
  canvas{display:block;width:100%;height:100%;cursor:grab}
  canvas:active{cursor:grabbing}
  #tafel{position:fixed;top:12px;left:12px;background:rgba(20,20,20,.86);padding:12px 14px;
         border-radius:8px;max-width:330px;backdrop-filter:blur(4px)}
  #tafel h1{font-size:14px;margin:0 0 6px}
  #tafel p{margin:2px 0;color:#aaa;font-size:12px}
  label{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:12px}
  input[type=range]{flex:1;accent-color:#7cc}
  .wert{min-width:52px;text-align:right;color:#7cc;font-variant-numeric:tabular-nums}
  button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:5px;
         padding:5px 9px;cursor:pointer;font-size:12px;margin-top:8px}
  button:hover{background:#363636}
  #hilfe{position:fixed;bottom:12px;left:12px;color:#777;font-size:11px}
</style></head><body>
<canvas id="c"></canvas>
<div id="tafel">
  <h1>__KOPF__</h1>
  <p>__ZEILE1__</p><p>__ZEILE2__</p>
  <label>Punktgroesse<input type="range" id="groesse" min="1" max="8" step="0.5" value="2.5">
    <span class="wert" id="groesse_w">2.5</span></label>
  <label>ab Hoehe<input type="range" id="schwelle" min="__ZMIN__" max="__ZMAX__" step="0.5" value="__ZMIN__">
    <span class="wert" id="schwelle_w">__ZMIN__ m</span></label>
  <label title="0 ist der geschaetzte Boden. Er ist im Bestand nicht sichtbar, deshalb beginnen die Punkte darueber.">Bodenebene<input type="range" id="ebene" min="0" max="__ZMIN__" step="0.5" value="__ZMIN__">
    <span class="wert" id="ebene_w">__ZMIN__ m</span></label>
  <button id="farbe">Farbe: nach Hoehe</button>
  <button id="boden">Boden ausblenden</button>
  <button id="zurueck">Ansicht zuruecksetzen</button>
</div>
<div id="hilfe">ziehen = drehen &nbsp;|&nbsp; Rad = zoomen &nbsp;|&nbsp; rechte Taste = verschieben
&nbsp;|&nbsp; Bodenebene auf 0 zeigt den geschaetzten Boden, den man im Bestand nicht sieht</div>
<script>
const DATEN = {
  xyz: "__XYZ__", rgb: "__RGB__", n: __N__,
  min: __MIN__, skala: __SKALA__, zmin: __ZMIN__, zmax: __ZMAX__,
  boden: __BODEN__, bild: "__BILD__"
};

function entpacke(b64, Typ) {
  const roh = atob(b64), puffer = new ArrayBuffer(roh.length), sicht = new Uint8Array(puffer);
  for (let i = 0; i < roh.length; i++) sicht[i] = roh.charCodeAt(i);
  return new Typ(puffer);
}

// --- Matrizen (Spaltenordnung wie in WebGL erwartet) ---
function perspektive(fovy, seite, nah, fern) {
  const f = 1 / Math.tan(fovy / 2), d = nah - fern;
  return [f/seite,0,0,0, 0,f,0,0, 0,0,(fern+nah)/d,-1, 0,0,2*fern*nah/d,0];
}
function blickRichtung(auge, ziel, oben) {
  let z = [auge[0]-ziel[0], auge[1]-ziel[1], auge[2]-ziel[2]];
  let l = Math.hypot(...z); z = z.map(v => v/l);
  let x = [oben[1]*z[2]-oben[2]*z[1], oben[2]*z[0]-oben[0]*z[2], oben[0]*z[1]-oben[1]*z[0]];
  l = Math.hypot(...x) || 1; x = x.map(v => v/l);
  const y = [z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]];
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
          -(x[0]*auge[0]+x[1]*auge[1]+x[2]*auge[2]),
          -(y[0]*auge[0]+y[1]*auge[1]+y[2]*auge[2]),
          -(z[0]*auge[0]+z[1]*auge[1]+z[2]*auge[2]), 1];
}
function malMatrix(a, b) {
  const r = new Array(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    for (let k = 0; k < 4; k++) r[i*4+j] += a[k*4+j] * b[i*4+k];
  return r;
}

const leinwand = document.getElementById("c");
const gl = leinwand.getContext("webgl", {antialias:true, alpha:false});
if (!gl) document.body.innerHTML = "<p style='padding:2em'>WebGL steht in diesem Browser nicht zur Verfuegung.</p>";

function bauen(art, quelle) {
  const s = gl.createShader(art); gl.shaderSource(s, quelle); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
  return s;
}
function programm(v, f) {
  const p = gl.createProgram();
  gl.attachShader(p, bauen(gl.VERTEX_SHADER, v));
  gl.attachShader(p, bauen(gl.FRAGMENT_SHADER, f));
  gl.linkProgram(p); return p;
}

const punktProg = programm(`
  attribute vec3 lage; attribute vec3 farbe;
  uniform mat4 mvp; uniform float groesse, zmin, zspanne, nachHoehe, schwelle;
  varying vec3 vFarbe; varying float vWeg;
  vec3 viridis(float t){                       // Naeherung der Viridis-Skala
    return clamp(vec3(
      0.28 + t*(-0.35 + t*(2.9 + t*(-2.1))),
      0.0  + t*(1.42 + t*(-0.86 + t*0.33)),
      0.33 + t*(1.26 + t*(-3.1 + t*1.9))), 0.0, 1.0);
  }
  void main(){
    float h = (lage.z - zmin) / max(zspanne, 0.001);
    vWeg = lage.z < schwelle ? 1.0 : 0.0;
    vFarbe = mix(farbe, viridis(clamp(h, 0.0, 1.0)), nachHoehe);
    gl_Position = mvp * vec4(lage, 1.0);
    gl_PointSize = groesse * (300.0 / max(gl_Position.w, 1.0));
  }`, `
  precision mediump float; varying vec3 vFarbe; varying float vWeg;
  void main(){
    if (vWeg > 0.5) discard;
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;             // runde Punkte
    gl_FragColor = vec4(vFarbe, 1.0);
  }`);

const bodenProg = programm(`
  attribute vec3 lage; attribute vec2 uv; uniform mat4 mvp; uniform float ebene;
  varying vec2 vUv;
  void main(){ vUv = uv; gl_Position = mvp * vec4(lage.xy, lage.z + ebene, 1.0); }`, `
  precision mediump float; uniform sampler2D bild; varying vec2 vUv;
  void main(){ gl_FragColor = vec4(texture2D(bild, vUv).rgb * 0.6, 1.0); }`);

const xyz = entpacke(DATEN.xyz, Int16Array), rgb = entpacke(DATEN.rgb, Uint8Array);
const lagePuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, lagePuffer);
const echt = new Float32Array(DATEN.n * 3);
for (let i = 0; i < DATEN.n * 3; i++) echt[i] = DATEN.min[i % 3] + xyz[i] * DATEN.skala;
gl.bufferData(gl.ARRAY_BUFFER, echt, gl.STATIC_DRAW);
const farbPuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, farbPuffer);
const farbF = new Float32Array(DATEN.n * 3);
for (let i = 0; i < DATEN.n * 3; i++) farbF[i] = rgb[i] / 255;
gl.bufferData(gl.ARRAY_BUFFER, farbF, gl.STATIC_DRAW);

let bodenPuffer = null, uvPuffer = null, textur = null;
if (DATEN.bild) {
  const b = DATEN.boden;
  bodenPuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bodenPuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    b[0],b[3],0, b[1],b[3],0, b[1],b[2],0, b[0],b[3],0, b[1],b[2],0, b[0],b[2],0]), gl.STATIC_DRAW);
  uvPuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, uvPuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1]), gl.STATIC_DRAW);
  textur = gl.createTexture();
  const im = new Image();
  im.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, textur);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, im);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    zeichnen();
  };
  im.src = "data:image/jpeg;base64," + DATEN.bild;
}

const mitte = [(DATEN.boden[0]+DATEN.boden[1])/2, (DATEN.boden[2]+DATEN.boden[3])/2,
               (DATEN.zmin+DATEN.zmax)/2];   // Mitte der Wolke, nicht der unsichtbare Boden
const weite = Math.max(DATEN.boden[1]-DATEN.boden[0], DATEN.boden[3]-DATEN.boden[2]);
const anfang = {azimut: -50, hoehe: 28, abstand: weite * 1.6, ziel: mitte.slice()};
let blick = JSON.parse(JSON.stringify(anfang));
let nachHoehe = 1, bodenAn = true, punktGroesse = 2.5, schwelle = DATEN.zmin;
// Vorgabe: die Ebene sitzt unter den tiefsten sichtbaren Punkten, sonst klafft
// eine Luecke -- der geschaetzte Boden bei 0 ist im Bestand ja nicht zu sehen.
let bodenEbene = DATEN.zmin;

function zeichnen() {
  const b = leinwand.clientWidth, h = leinwand.clientHeight;
  if (leinwand.width !== b || leinwand.height !== h) { leinwand.width = b; leinwand.height = h; }
  gl.viewport(0, 0, b, h);
  gl.clearColor(0.07, 0.07, 0.08, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  const az = blick.azimut * Math.PI/180, el = blick.hoehe * Math.PI/180;
  const auge = [blick.ziel[0] + blick.abstand*Math.cos(el)*Math.cos(az),
                blick.ziel[1] + blick.abstand*Math.cos(el)*Math.sin(az),
                blick.ziel[2] + blick.abstand*Math.sin(el)];
  const mvp = malMatrix(perspektive(Math.PI/4, b/h, weite*0.02, weite*20),
                        blickRichtung(auge, blick.ziel, [0,0,1]));

  if (bodenAn && textur && bodenPuffer) {
    gl.useProgram(bodenProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(bodenProg, "mvp"), false, new Float32Array(mvp));
    const aL = gl.getAttribLocation(bodenProg, "lage"), aU = gl.getAttribLocation(bodenProg, "uv");
    gl.bindBuffer(gl.ARRAY_BUFFER, bodenPuffer);
    gl.enableVertexAttribArray(aL); gl.vertexAttribPointer(aL, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, uvPuffer);
    gl.enableVertexAttribArray(aU); gl.vertexAttribPointer(aU, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, textur);
    gl.uniform1i(gl.getUniformLocation(bodenProg, "bild"), 0);
    gl.uniform1f(gl.getUniformLocation(bodenProg, "ebene"), bodenEbene);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(aU);
  }

  gl.useProgram(punktProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(punktProg, "mvp"), false, new Float32Array(mvp));
  gl.uniform1f(gl.getUniformLocation(punktProg, "groesse"), punktGroesse);
  gl.uniform1f(gl.getUniformLocation(punktProg, "zmin"), DATEN.zmin);
  gl.uniform1f(gl.getUniformLocation(punktProg, "zspanne"), DATEN.zmax - DATEN.zmin);
  gl.uniform1f(gl.getUniformLocation(punktProg, "nachHoehe"), nachHoehe);
  gl.uniform1f(gl.getUniformLocation(punktProg, "schwelle"), schwelle);
  const pL = gl.getAttribLocation(punktProg, "lage"), pF = gl.getAttribLocation(punktProg, "farbe");
  gl.bindBuffer(gl.ARRAY_BUFFER, lagePuffer);
  gl.enableVertexAttribArray(pL); gl.vertexAttribPointer(pL, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, farbPuffer);
  gl.enableVertexAttribArray(pF); gl.vertexAttribPointer(pF, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, DATEN.n);
}

let zieht = false, verschiebt = false, letzte = [0, 0];
leinwand.addEventListener("mousedown", e => {
  zieht = e.button === 0; verschiebt = e.button === 2; letzte = [e.clientX, e.clientY];
});
window.addEventListener("mouseup", () => { zieht = verschiebt = false; });
window.addEventListener("mousemove", e => {
  const dx = e.clientX - letzte[0], dy = e.clientY - letzte[1];
  letzte = [e.clientX, e.clientY];
  if (zieht) {
    blick.azimut -= dx * 0.4;
    blick.hoehe = Math.max(-85, Math.min(89, blick.hoehe + dy * 0.3));
    zeichnen();
  } else if (verschiebt) {
    const az = blick.azimut * Math.PI/180, s = blick.abstand * 0.0015;
    blick.ziel[0] += (Math.sin(az)*dx + Math.cos(az)*dy) * s;
    blick.ziel[1] += (-Math.cos(az)*dx + Math.sin(az)*dy) * s;
    zeichnen();
  }
});
leinwand.addEventListener("contextmenu", e => e.preventDefault());
leinwand.addEventListener("wheel", e => {
  e.preventDefault();
  blick.abstand = Math.max(weite*0.05, Math.min(weite*8, blick.abstand * (1 + Math.sign(e.deltaY)*0.12)));
  zeichnen();
}, {passive:false});
window.addEventListener("resize", zeichnen);

const g = document.getElementById("groesse"), s = document.getElementById("schwelle");
const eb = document.getElementById("ebene");
eb.oninput = () => {
  bodenEbene = +eb.value;
  document.getElementById("ebene_w").textContent = (+eb.value).toFixed(1) + " m";
  zeichnen();
};
g.oninput = () => { punktGroesse = +g.value; document.getElementById("groesse_w").textContent = g.value; zeichnen(); };
s.oninput = () => { schwelle = +s.value; document.getElementById("schwelle_w").textContent = (+s.value).toFixed(1)+" m"; zeichnen(); };
document.getElementById("farbe").onclick = e => {
  nachHoehe = nachHoehe > 0.5 ? 0 : 1;
  e.target.textContent = "Farbe: " + (nachHoehe ? "nach Hoehe" : "Originalbild"); zeichnen();
};
document.getElementById("boden").onclick = e => {
  bodenAn = !bodenAn; e.target.textContent = bodenAn ? "Boden ausblenden" : "Boden einblenden"; zeichnen();
};
document.getElementById("zurueck").onclick = () => { blick = JSON.parse(JSON.stringify(anfang)); zeichnen(); };
zeichnen();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--karten", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_karten"))
    parser.add_argument("--ft", type=Path,
                        default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--modellart", default="tiefe", choices=("tiefe", "hoehe"),
                        help="tiefe: via the depth, with a terrain model. hoehe: outputs metres "
                             "directly, but does not distinguish between stands.")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_3d"))
    parser.add_argument("--frames", nargs="*", default=None, metavar="ORDNER/DATEI")
    parser.add_argument("--hfov-deg", type=float, default=48.0)
    parser.add_argument("--schritt", type=int, default=3, help="Every n-th pixel.")
    parser.add_argument("--max-punkte", type=int, default=260000,
                        help="Upper bound; above it points are thinned at random.")
    parser.add_argument("--max-neigung", type=float, default=8.0)
    parser.add_argument("--min-hoehe", type=float, default=0.5)
    parser.add_argument("--boden-breite", type=int, default=1400, help="Resolution of the ground texture.")
    parser.add_argument("--ohne-boden", action="store_true")
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE",
                        default=["dense=51", "dense1=69", "mixed=92", "mixed1=103",
                                 "pines=60", "urban=120"])
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--kachel-m", type=float, default=35.0)
    parser.add_argument("--boden-faktor", type=float, default=0.917)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    vorgaben = {e.split("=")[0]: float(e.split("=")[1]) for e in args.altitudes}
    k = inferenz.k_von_fov(args.hfov_deg)
    rng = np.random.default_rng(0)

    ordner = sorted(p for p in args.input.iterdir() if p.is_dir())
    if args.frames:
        auswahl = [args.input / f for f in args.frames]
    else:
        auswahl = []
        for o in ordner:
            treffer = sorted(f for f in o.iterdir() if f.suffix.lower() in BILDENDUNGEN)
            if treffer:
                auswahl.append(treffer[0])

    model = None
    for pfad in auswahl:
        ordnername = pfad.parent.name
        name = f"{ordnername}_{pfad.stem}"
        frame_bgr = cv2.imread(str(pfad))
        bild_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hinweis = kamera_pruefen(bild_rgb, pfad)
        if hinweis:
            print(f"  ACHTUNG {hinweis}", flush=True)

        H = vorgaben.get(ordnername)
        if H is None:
            m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", ordnername, re.IGNORECASE)
            H = float(m.group(1)) if m else args.altitude

        import torch
        if model is None:
            device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
            model = inferenz.lade(str(args.ft), device, fov_head=False)
        tiefe, hoehe_direkt = inferenz.karten_von(model, bild_rgb, art=args.modellart, k=k,
                                                  flughoehe_m=H, device=device)

        h, w = tiefe.shape
        f_px = k * w
        gsd = H / f_px
        # With the height model the ground reference is already in the prediction.
        boden = (tiefe + hoehe_direkt if hoehe_direkt is not None
                 else bodenmodell(tiefe, gsd, args.kachel_m, 97.0, args.boden_faktor))

        s = max(1, args.schritt)
        v, u = np.mgrid[0:h:s, 0:w:s].astype(np.float32)
        d = tiefe[::s, ::s]
        X = (u - (w - 1) / 2.0) * d / f_px
        Y = -(v - (h - 1) / 2.0) * d / f_px
        Z = boden[::s, ::s] - d
        xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        rgb = bild_rgb[::s, ::s].reshape(-1, 3).astype(np.uint8)

        behalten = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= args.min_hoehe)
        if args.max_neigung > 0:
            behalten &= kantenmaske(tiefe, gsd, args.max_neigung)[::s, ::s].ravel()
        xyz, rgb = xyz[behalten], rgb[behalten]
        if len(xyz) > args.max_punkte:
            wahl = rng.choice(len(xyz), args.max_punkte, replace=False)
            xyz, rgb = xyz[wahl], rgb[wahl]
        if len(xyz) < 100:
            print(f"  {name}: zu wenige Punkte", flush=True)
            continue

        # int16 instead of float32: half the file size, step size in millimetres.
        mins = xyz.min(axis=0)
        spanne = float(np.max(xyz.max(axis=0) - mins))
        skala = spanne / 32000.0
        ganz = np.clip(np.round((xyz - mins) / skala), -32768, 32767).astype("<i2")

        zmin, zmax = float(np.percentile(xyz[:, 2], 1)), float(np.percentile(xyz[:, 2], 99))
        bodenrand = [float(xyz[:, 0].min()), float(xyz[:, 0].max()),
                     float(xyz[:, 1].min()), float(xyz[:, 1].max())]

        bild_b64 = ""
        if not args.ohne_boden:
            faktor = args.boden_breite / frame_bgr.shape[1]
            klein = cv2.resize(frame_bgr, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_AREA)
            _, puffer = cv2.imencode(".jpg", klein, [cv2.IMWRITE_JPEG_QUALITY, 82])
            bild_b64 = base64.b64encode(puffer.tobytes()).decode("ascii")

        html = (VORLAGE
                .replace("__TITEL__", f"{name} — Punktwolke")
                .replace("__KOPF__", f"{ordnername}/{pfad.name}")
                .replace("__ZEILE1__", f"{len(xyz)} Punkte &middot; Hoehe ueber Boden "
                                       f"{zmin:.1f} bis {zmax:.1f} m")
                .replace("__ZEILE2__", f"Flughoehe {H:.0f} m &middot; Bildwinkel {args.hfov_deg:.0f}&deg; "
                                       f"&middot; Flaeche {np.ptp(xyz[:, 0]):.0f} &times; "
                                       f"{np.ptp(xyz[:, 1]):.0f} m")
                .replace("__XYZ__", base64.b64encode(ganz.tobytes()).decode("ascii"))
                .replace("__RGB__", base64.b64encode(rgb.tobytes()).decode("ascii"))
                .replace("__N__", str(len(xyz)))
                .replace("__MIN__", "[" + ",".join(f"{v:.4f}" for v in mins) + "]")
                .replace("__SKALA__", f"{skala:.8f}")
                .replace("__ZMIN__", f"{zmin:.1f}")
                .replace("__ZMAX__", f"{zmax:.1f}")
                .replace("__BODEN__", "[" + ",".join(f"{v:.3f}" for v in bodenrand) + "]")
                .replace("__BILD__", bild_b64))

        ziel = args.out / f"viewer_{name}.html"
        ziel.write_text(html, encoding="utf-8")
        print(f"  {ziel.name:44s} {len(xyz):7d} Punkte, {ziel.stat().st_size/1e6:5.1f} MB", flush=True)

    print(f"\n-> {args.out}\nIm Browser oeffnen: ziehen dreht, Rad zoomt, rechte Taste verschiebt.")


if __name__ == "__main__":
    main()
