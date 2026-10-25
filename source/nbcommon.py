import nbformat as nbf
def M(s): return ("md", s.strip("\n"))
def C(s, hidden=False): return ("code", s.strip("\n"), hidden)
def build(cells, path):
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"},
                  "colab": {"provenance": [], "toc_visible": True}}
    out = []
    for c in cells:
        if c[0] == "md":
            out.append(nbf.v4.new_markdown_cell(c[1]))
        else:
            cell = nbf.v4.new_code_cell(c[1])
            if c[2]: cell.metadata["cellView"] = "form"
            out.append(cell)
    n.cells = out
    nbf.write(n, path)

HEADER = """> **Aquatic Guardian · Learn computer vision** — Module {num} of 09.  
> **Time:** {time}. **Runs on:** free Google Colab, CPU is enough. **Skip if:** {skip}

**How to use this notebook.** In Colab choose **File → Save a copy in Drive** first, so your work is kept.
Then **Runtime → Run all**. Every cell runs without errors before you touch anything; the exercises print
feedback rather than crashing. Cells are marked:

- ▶ **Run and read** — nothing to change.
- ✏️ **Predict, then change** — write down what you expect, change one value, re-run.
- 🧩 **Your turn** — replace `None` with your code. A checker tells you whether it works. Hidden solutions follow.
"""

SETUP = r'''#@title ▶ Setup: scenes and real wildlife photographs (run it; you don't need to read it yet)
import numpy as np, matplotlib.pyplot as plt, urllib.request
from pathlib import Path
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 10})

PHOTO_BASE = "https://raw.githubusercontent.com/ejri/aquatic-guardian-learning-modules/main/images/"
PHOTO_FILES = {
    "reef-school": "reef-school.jpg", "manatees": "manatees.jpg", "sea-lion": "sea-lion.jpg",
    "nurse-shark": "nurse-shark.jpg", "eagle-ray": "eagle-ray.jpg", "grouper": "grouper.jpg",
    "angelfish": "angelfish.jpg", "turtle": "turtle.jpg", "noaa-honu": "noaa-honu.jpg",
    "noaa-whale": "noaa-whale.jpg", "kelp-forest": "kelp-forest.jpg", "zebra-shark": "zebra-shark.jpg",
}

def load_photo(name, max_side=480):
    """Load a real wildlife JPEG. Looks in ./images and ../images, then downloads from GitHub."""
    file = PHOTO_FILES.get(name, name if str(name).endswith(".jpg") else str(name) + ".jpg")
    arr = None
    for candidate in (Path("images") / file, Path("../images") / file, Path(file)):
        if candidate.is_file():
            arr = plt.imread(candidate)
            break
    if arr is None:
        urllib.request.urlretrieve(PHOTO_BASE + file, file)
        arr = plt.imread(file)
    if getattr(arr, "dtype", None) == np.uint8:
        arr = arr.astype(float) / 255.0
    arr = np.asarray(arr[..., :3], dtype=float)
    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        yy = (np.linspace(0, h - 1, max(1, int(h * scale)))).astype(int)
        xx = (np.linspace(0, w - 1, max(1, int(w * scale)))).astype(int)
        arr = arr[yy][:, xx]
    return arr


# Illustrative attenuation coefficients beta (1/m) for R, G, B, and the colour of the veiling light.
# These are round teaching numbers with the right ordering, not measurements of any real site.
WATER = {
    "clear_ocean":    dict(beta=(0.40, 0.07, 0.04), veil=(0.02, 0.20, 0.42)),
    "coastal":        dict(beta=(0.55, 0.20, 0.30), veil=(0.05, 0.32, 0.27)),
    "turbid_harbour": dict(beta=(0.95, 0.60, 0.75), veil=(0.14, 0.30, 0.22)),
}

def _fish_mask(H, W, cx, cy, L, ang, aspect=4.0):
    """Ellipse body plus a triangular tail. L is total length (snout to tail tip) in pixels."""
    yy, xx = np.mgrid[0:H, 0:W]
    a = L / 2.25; b = a / aspect
    X = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    Y = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    body = (X / a) ** 2 + (Y / b) ** 2 <= 1
    tx = -X - 0.80 * a
    tail = (tx >= 0) & (tx <= 0.45 * a) & (np.abs(Y) <= 1.3 * b * tx / (0.45 * a))
    return body | tail, X, Y, a, b

FISH_COLOURS = [(0.97, 0.55, 0.12), (0.96, 0.84, 0.22), (0.80, 0.82, 0.84), (0.93, 0.35, 0.25)]

def make_scene(n_fish=4, H=120, W=180, seed=0, bg_seed=None, fish_dist=(2.0, 6.0), bg_dist=12.0,
               length_px=(22, 40), min_gap=0.55, touching=False, aspect=(4.0, 4.0)):
    """Returns a dict with the true scene radiance J (what you'd see in air), a distance map in metres,
    one boolean mask and one box (x1, y1, x2, y2) per fish, and per-fish truth."""
    r = np.random.default_rng(seed)
    rb = np.random.default_rng(seed if bg_seed is None else bg_seed)
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    J = np.stack([0.48 + 0.10 * yy, 0.43 + 0.08 * yy, 0.34 + 0.04 * xx], -1)       # reef wall / sand
    J = J + 0.04 * rb.standard_normal((H, W, 1)) + 0.015 * rb.standard_normal((H, W, 3))
    dist = np.full((H, W), float(bg_dist))
    masks, boxes, truth, placed = [], [], [], []
    for i in range(n_fish):
        for _ in range(200):
            L = r.uniform(*length_px); ang = r.uniform(-0.6, 0.6) + (np.pi if r.random() < 0.5 else 0)
            cx = r.uniform(0.15 * W, 0.85 * W); cy = r.uniform(0.18 * H, 0.82 * H)
            gap = 0.25 if touching else min_gap
            if all((cx - px) ** 2 + (cy - py) ** 2 > (gap * (L + pl)) ** 2 for px, py, pl in placed):
                break
        placed.append((cx, cy, L))
        asp = aspect[0] if aspect[0] == aspect[1] else r.uniform(*aspect)
        m, X, Y, a, b = _fish_mask(H, W, cx, cy, L, ang, asp)
        col = np.array(FISH_COLOURS[i % len(FISH_COLOURS)])
        J[m] = col
        stripe = m & (np.abs(X - 0.15 * a) < 0.09 * a)
        J[stripe] = (0.95, 0.95, 0.92)
        eye = (X - 0.62 * a) ** 2 + (Y - 0.15 * b) ** 2 < (0.12 * b + 0.6) ** 2
        J[eye & m] = (0.05, 0.05, 0.08)
        d = r.uniform(*fish_dist); dist[m] = d
        ys, xs = np.where(m)
        masks.append(m); boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        truth.append(dict(cx=float(cx), cy=float(cy), length_px=float(L), angle=float(ang), dist_m=float(d), aspect=float(asp)))
    return dict(J=np.clip(J, 0, 1), dist=dist, masks=masks, boxes=boxes, truth=truth)

def underwater(J, dist, water="coastal", light=1.0, photons=None, seed=0):
    """Simple image-formation model, per colour channel c:
         I_c = light * ( J_c * t_c  +  B_c * (1 - t_c) ),   t_c = exp(-beta_c * distance)
    photons: if set, adds shot noise (fewer photons -> noisier). Lower `light` means fewer photons."""
    beta = np.array(WATER[water]["beta"]); B = np.array(WATER[water]["veil"])
    t = np.exp(-beta[None, None, :] * dist[..., None])
    I = light * (J * t + B * (1 - t))
    if photons:
        rng = np.random.default_rng(seed)
        n = photons * light
        I = rng.poisson(np.clip(I, 0, None) / max(light, 1e-6) * n) / n * light
    return np.clip(I, 0, 1)

def degrade(img, turbidity=0.0, low_light=0.0, blur_k=1):
    """The degradation utility used in the Aquatic Guardian project notebooks (copied verbatim in spirit)."""
    out = img.copy()
    if blur_k > 1:
        pad = blur_k // 2; P = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge"); acc = np.zeros_like(out)
        for dy in range(blur_k):
            for dx in range(blur_k): acc += P[dy:dy + out.shape[0], dx:dx + out.shape[1], :]
        out = acc / (blur_k * blur_k)
    if turbidity > 0:
        veil = np.array([0.20, 0.42, 0.45]); out = (1 - turbidity) * out + turbidity * veil
        out = (1 - 0.5 * turbidity) * out + 0.5 * turbidity * out.mean()
    if low_light > 0: out = (out * (1 - 0.85 * low_light)) ** (1 + 0.6 * low_light)
    return np.clip(out, 0, 1)

def show(images, titles=None, size=3.2, cmap=None):
    images = images if isinstance(images, (list, tuple)) else [images]
    fig, ax = plt.subplots(1, len(images), figsize=(size * len(images), size * 0.72))
    ax = np.atleast_1d(ax)
    for i, (a, im) in enumerate(zip(ax, images)):
        a.imshow(im, cmap=cmap if im.ndim == 2 else None, vmin=0 if im.ndim == 2 and cmap else None,
                 vmax=1 if im.ndim == 2 and cmap else None)
        a.set_title(titles[i] if titles else ""); a.axis("off")
    plt.tight_layout(); plt.show()

def draw_boxes(img, boxes, colour=(1.0, 0.85, 0.0)):
    out = img.copy()
    for (x1, y1, x2, y2) in boxes:
        x2 -= 1; y2 -= 1
        out[y1, x1:x2 + 1] = colour; out[y2, x1:x2 + 1] = colour
        out[y1:y2 + 1, x1] = colour; out[y1:y2 + 1, x2] = colour
    return out
print("Setup ready. Water types:", list(WATER))'''

FRAMES = r'''#@title ▶ Setup: a staged dataset of short camera clips (fish / non_fish / empty_frame)
CLASSES = ["fish", "non_fish", "empty_frame"]     # Mentee 1's Level-1 classes
HW = 48                                           # frame size in pixels
HABITATS = {   # background colour in air, water type, typical object distance (m)
    "reef":    dict(bg=(0.52, 0.47, 0.36), water="clear_ocean",    dist=(1.0, 4.0)),
    "kelp":    dict(bg=(0.30, 0.34, 0.22), water="coastal",        dist=(1.0, 3.5)),
    "harbour": dict(bg=(0.36, 0.33, 0.28), water="turbid_harbour", dist=(0.6, 2.2)),
}

def new_session(cls, habitat, r):
    h = HABITATS[habitat]
    return dict(cls=cls, habitat=habitat, water=h["water"], bg=np.array(h["bg"]) + r.normal(0, 0.03, 3),
                obj_dist=r.uniform(*h["dist"]), bg_dist=r.uniform(6, 12), photons=r.uniform(250, 900),
                x0=r.uniform(14, 34), y0=r.uniform(14, 34), vx=r.normal(0, 0.6), vy=r.normal(0, 0.3),
                L=r.uniform(16, 30), ang=r.uniform(0, np.pi), aspect=r.uniform(3.2, 4.5),
                col=np.array(FISH_COLOURS[r.integers(len(FISH_COLOURS))]) + r.normal(0, 0.05, 3),
                blob_col=r.uniform(0.25, 0.6, 3), blob_r=r.uniform(4, 8), blob_aspect=r.uniform(1.0, 1.5))

def render_frame(s, i, r):
    yy, xx = np.mgrid[0:HW, 0:HW] / HW
    J = s["bg"] + 0.06 * np.stack([yy, 0.8 * yy, 0.5 * xx], -1) + 0.03 * r.standard_normal((HW, HW, 1))
    dist = np.full((HW, HW), s["bg_dist"])
    cx = np.clip(s["x0"] + s["vx"] * i, 8, HW - 8); cy = np.clip(s["y0"] + s["vy"] * i, 8, HW - 8)
    if s["cls"] == "fish":
        m = _fish_mask(HW, HW, cx, cy, s["L"], s["ang"] + 0.02 * i, s["aspect"])[0]; J[m] = s["col"]; dist[m] = s["obj_dist"]
    elif s["cls"] == "non_fish":     # crab, rock, jellyfish: compact blobs
        m = _fish_mask(HW, HW, cx, cy, 2.25 * s["blob_r"], s["ang"], s["blob_aspect"])[0]
        m &= ((np.mgrid[0:HW, 0:HW][1] - cx) ** 2 + (np.mgrid[0:HW, 0:HW][0] - cy) ** 2) <= (1.2 * s["blob_r"]) ** 2
        J[m] = s["blob_col"]; dist[m] = s["obj_dist"]
    return underwater(np.clip(J, 0, 1), dist, s["water"], photons=s["photons"], seed=int(r.integers(1 << 30)))

def make_clips(n_sessions_per_class=40, frames_per_clip=12, habitats=("reef", "kelp", "harbour"), seed=0):
    """Each session is one short clip: one object (or none) drifting past a fixed camera.
    Returns images X (N, HW, HW, 3), labels y (N,), session ids (N,), habitat names (N,)."""
    r = np.random.default_rng(seed); X, y, sess, hab = [], [], [], []
    sid = 0
    for ci, cls in enumerate(CLASSES):
        for k in range(n_sessions_per_class):
            habitat = habitats[k % len(habitats)]
            s = new_session(cls, habitat, r)
            for i in range(frames_per_clip):
                X.append(render_frame(s, i, r)); y.append(ci); sess.append(sid); hab.append(habitat)
            sid += 1
    return np.array(X), np.array(y), np.array(sess), np.array(hab)
print("Clip generator ready. Classes:", CLASSES)'''
