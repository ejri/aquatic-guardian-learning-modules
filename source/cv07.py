from nbcommon import *
cells = [
M("# Module 07 — Detection, measurement and calibration\n\n" + HEADER.format(num="07", time="about 2.5 hours",
  skip="you can compute IoU, NMS and average precision, and you know how a flat underwater port biases size estimates.") + """
### By the end you can
1. Represent boxes, compute **IoU**, and remove duplicates with **non-maximum suppression**.
2. Evaluate a detector with a **precision–recall curve** and **average precision (AP)**.
3. Convert pixels to centimetres, and explain why the scale depends on distance.
4. Quantify the size bias from a **flat underwater camera port** (refraction), and correct it to first order.
5. Propagate length error into mass error with $W = aL^b$, and spot a circular welfare metric.

This module covers **Mentee 2, Tasks 2 and 6–9**, and the measurement side of the project's open item on
flat-port refraction correction.
"""),
C(SETUP, hidden=True),
C('''from scipy import ndimage
import pandas as pd
def iou(a, b):
    """a, b: boxes (x1, y1, x2, y2) with x2, y2 exclusive."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
print("IoU of half-overlapping boxes:", round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 3), "(1/3, not 1/2)")'''),
M("""## 1. Boxes and IoU ▶

**Intersection over union** = overlap area ÷ combined area. Detectors are usually scored as correct when IoU with a
true box is at least 0.5. Note that two boxes overlapping by half have IoU of only 1/3.

Formats differ between tools: `(x1, y1, x2, y2)` corners, `(cx, cy, w, h)` centre and size (YOLO, often divided by
the image size), or `(x, y, w, h)` (COCO). Convert explicitly; silent format mix-ups are a classic bug."""),
C('''def xyxy_to_cxcywh(b): x1, y1, x2, y2 = b; return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
def cxcywh_to_xyxy(b): cx, cy, w, h = b; return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
b = (20, 30, 60, 50); print(b, "->", xyxy_to_cxcywh(b), "->", cxcywh_to_xyxy(xyxy_to_cxcywh(b)))'''),
M("""## 2. Non-maximum suppression ▶

Detectors often fire several overlapping boxes on one animal. NMS keeps the highest-scoring box and deletes any
other box that overlaps it by more than a threshold, then repeats."""),
C('''def nms(boxes, scores, thr=0.5):
    order = list(np.argsort(scores)[::-1]); keep = []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < thr]
    return keep
r = np.random.default_rng(0)
scene = make_scene(n_fish=3, seed=21); img = underwater(scene["J"], scene["dist"], "clear_ocean")
cand, sc = [], []
for (x1, y1, x2, y2) in scene["boxes"]:
    for _ in range(5):
        j = r.integers(-4, 5, 4); cand.append((x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3])); sc.append(r.uniform(0.5, 1))
keep = nms(cand, np.array(sc))
show([draw_boxes(img, cand), draw_boxes(img, [cand[k] for k in keep])], [f"{len(cand)} raw boxes", f"after NMS: {len(keep)}"])'''),
M("""## 3. Scoring a detector: precision–recall and AP ▶

A detector outputs boxes with confidence scores. Sort all detections by score; walk down the list, counting a
**true positive** when a detection matches an unmatched true box at IoU ≥ 0.5. At each step you have a precision
and a recall. **Average precision** is the area under that curve (with the standard "envelope" smoothing).
**mAP** is AP averaged over classes, and sometimes over IoU thresholds (e.g. "mAP@0.5:0.95").

Our detector is the classical pipeline from Modules 01 and 03, with each blob's mean contrast as its score."""),
C('''def classical_detector(img, thr=0.05, min_area=25):
    border = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3), img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
    diff = np.sqrt(((img - np.median(border, 0)) ** 2).sum(-1))
    lab, n = ndimage.label(ndimage.binary_opening(diff > thr))
    boxes, scores = [], []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        region = lab[sl] == k
        if region.sum() >= min_area:
            boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)); scores.append(diff[sl][region].mean())
    return boxes, np.array(scores)

def evaluate(scenes, water, iou_thr=0.5):
    all_scores, all_tp, n_gt = [], [], 0
    for sc_ in scenes:
        im = underwater(sc_["J"], sc_["dist"], water, photons=500, seed=1)
        boxes, scores = classical_detector(im); gt = sc_["boxes"]; n_gt += len(gt); used = set()
        for i in np.argsort(-scores):
            ious = [iou(boxes[i], g) if j not in used else 0 for j, g in enumerate(gt)]
            j = int(np.argmax(ious)) if ious else -1
            hit = j >= 0 and ious[j] >= iou_thr
            if hit: used.add(j)
            all_scores.append(scores[i]); all_tp.append(1 if hit else 0)
    return np.array(all_scores), np.array(all_tp), n_gt

def average_precision(scores, tp, n_gt):
    order = np.argsort(-scores); tp = tp[order]; ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    recall = ctp / n_gt; precision = ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0], recall, [1]]); mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), recall, precision

scenes = [make_scene(n_fish=5, seed=200 + i, fish_dist=(1.5, 9.0)) for i in range(30)]
fig, ax = plt.subplots(figsize=(5.5, 3.8)); rows = []
for water in WATER:
    s, t, n = evaluate(scenes, water); ap, rec, prec = average_precision(s, t, n)
    ax.plot(rec, prec, lw=2, label=f"{water}: AP = {ap:.2f}"); rows.append(dict(water=water, AP50=ap, max_recall=rec.max() if len(rec) else 0))
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
ax.set_title("Same detector, same fish, three waters"); plt.tight_layout(); plt.show()
pd.DataFrame(rows).set_index("water").round(3)'''),
M("""✏️ **Interpret:** in turbid water the curve stops early — **maximum recall** is far below 1. No threshold choice
recovers fish whose contrast fell below the detector's sensitivity (Module 02). That ceiling is the
**detection probability** that Module 09 corrects for.

## 4. Pretrained detectors: check the label set first

Off-the-shelf detectors are trained on COCO's 80 everyday categories. Is "fish" one of them?"""),
C('''try:
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
    cats = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT.meta["categories"]
    animals = [c for c in cats if c in ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")]
    print("COCO animal classes:", animals); print("'fish' in COCO:", "fish" in cats)
except Exception as e:
    print("torchvision not available here:", type(e).__name__)'''),
M("""There is no fish class, so a COCO detector must be **fine-tuned** on underwater data before it's useful. Options
in Colab: torchvision's Faster R-CNN or RetinaNet (BSD licence), YOLOX (Apache-2.0), or Ultralytics YOLO
(**AGPL-3.0**, which also covers models you train with it — record that before adopting it, as the Weeks 1–4
scaffold notes). The evaluation code above works unchanged for any of them.

## 5. From pixels to centimetres ▶

A simple (pinhole) camera maps an object of length $L$ at distance $Z$ to an image length $\\ell = fL/Z$, where $f$
is the focal length in pixels. So **pixels per centimetre depends on distance**. A calibration target only gives the
right scale for objects at the target's distance. A fish half a metre closer looks bigger and is over-estimated."""),
C('''f_px = 900.0                    # focal length in pixels (from calibration)
calib_Z = 1.0                    # calibration target placed 1.0 m from the camera
scale_px_per_cm = f_px / (calib_Z * 100)
for Z in (0.6, 0.8, 1.0, 1.2, 1.5):
    L_true = 30.0; ell = f_px * (L_true / 100) / Z
    print(f"fish at {Z:.1f} m: {ell:6.1f} px -> estimated {ell / scale_px_per_cm:5.1f} cm (true 30.0, error {100 * (ell / scale_px_per_cm - 30) / 30:+.0f}%)")'''),
M(r"""Fixes: keep fish in a known plane (a channel or tank wall), use **stereo** cameras, or use **laser scalers**
(two parallel lasers a known distance apart) visible on the fish.

## 6. The flat port: refraction makes fish look bigger ▶

An underwater camera looks out through a flat window. Light bends at the water–air boundary (Snell's law,
$n_w \sin\theta_w = n_a \sin\theta_a$ with $n_w \approx 1.33$). Near the centre, everything looks about **1.33× larger**
than the in-air pinhole model predicts, and the effect **grows toward the edges** of the frame. The field of view
also shrinks: no ray beyond about 48.8° in water can reach the camera.

Assumptions in this sketch: thin port, camera axis perpendicular to the port, glass thickness ignored."""),
C('''n_w = 1.33
def x_flatport(X, Z, f=1.0):
    th_w = np.arctan2(X, Z); return f * np.tan(np.arcsin(np.clip(n_w * np.sin(th_w), -1, 1)))
def x_air(X, Z, f=1.0): return f * X / Z

angles = np.radians(np.linspace(0.5, 45, 90))
mag = x_flatport(np.tan(angles), 1.0) / x_air(np.tan(angles), 1.0)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(np.degrees(angles), mag, lw=2); ax[0].axhline(n_w, ls="--", color="k", lw=1)
ax[0].set_xlabel("angle off the camera axis, in water (deg)"); ax[0].set_ylabel("apparent magnification"); ax[0].set_title("Flat-port magnification vs in-air model")

Z, L = 1.0, 0.30                                   # fish 30 cm long, 1 m away
centres = np.linspace(0, 0.7, 50)                  # lateral position of the fish's centre (m)
ell = x_flatport(centres + L / 2, Z) - x_flatport(centres - L / 2, Z)
est_air = ell * Z / 1.0; est_parax = est_air / n_w
ax[1].plot(np.degrees(np.arctan(centres / Z)), 100 * (est_air - L) / L, lw=2, label="calibrated in air")
ax[1].plot(np.degrees(np.arctan(centres / Z)), 100 * (est_parax - L) / L, lw=2, label="in-air ÷ 1.33 (paraxial fix)")
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("fish position off-axis (deg)"); ax[1].set_ylabel("length error (%)")
ax[1].legend(fontsize=8); ax[1].set_title("Length error for a 30 cm fish at 1 m"); plt.tight_layout(); plt.show()'''),
M("""✏️ **Interpret:** the in-air calibration over-estimates every fish by a third or more. Dividing by 1.33 fixes the
centre of the frame but not the edges. Practical options: calibrate **underwater** with a checkerboard at working
distance (this absorbs most of the effect into the focal length and lens distortion), restrict measurements to the
central part of the frame, use a **dome port**, or use a full refractive camera model.

## 7. From length to mass: errors get multiplied ▶

Fish mass is usually estimated from length with $W = aL^b$, with $b$ typically near 3 (species values from FishBase).
Then a relative length error $\\varepsilon$ becomes a relative mass error of about $b\\,\\varepsilon$: a 5% length error
turns into roughly 15% mass error."""),
C('''a, b_ = 0.012, 3.0
L = np.random.default_rng(0).normal(25, 4, 5000)
for eps in (0.02, 0.05, 0.10, 0.33):
    W_true = a * L ** b_; W_est = a * (L * (1 + eps)) ** b_
    print(f"length error {100 * eps:4.0f}%  ->  mass error {100 * np.mean(W_est / W_true - 1):5.1f}%")'''),
M(r"""## 8. Spot the problem: a welfare metric that can't move ▶

**Fulton's condition factor** $K = 100\,W/L^3$ compares a fish's mass with what its length predicts; low $K$ can flag a
thin, unwell fish. Now suppose mass isn't measured, but *estimated from length* with $W = aL^b$:

$$K = \frac{100\,aL^b}{L^3} = 100\,a\,L^{\,b-3}.$$

With $b = 3$, $K = 100a$ for **every** fish, however thin or fat it really is."""),
C('''def measure(lab, k):
    ys, xs = np.where(lab == k)
    if len(xs) < 20: return None
    P = np.stack([xs - xs.mean(), ys - ys.mean()]); ev, evec = np.linalg.eigh(np.cov(P))
    along, across = P.T @ evec[:, np.argmax(ev)], P.T @ evec[:, np.argmin(ev)]
    L = np.ptp(along); mid = (along.max() + along.min()) / 2
    body = np.abs(along - mid) < 0.2 * L                        # central fifth of the length: skips the tail fin
    return dict(cx=xs.mean(), cy=ys.mean(), L_px=L, depth_px=np.ptp(across[body]))

SCALE = 4.0                                       # px per cm, assumed correct here
rows = []
for i in range(25):
    sc_ = make_scene(n_fish=4, seed=300 + i, aspect=(3.0, 5.2), H=220, W=340, length_px=(50, 70), min_gap=0.7)
    dist = sc_["dist"].copy(); dist[np.any(sc_["masks"], 0)] = 1.0          # every fish 1 m away, clear water
    im = underwater(sc_["J"], dist, "clear_ocean")
    border = np.concatenate([im[:4].reshape(-1, 3), im[-4:].reshape(-1, 3), im[:, :4].reshape(-1, 3), im[:, -4:].reshape(-1, 3)])
    diff = np.sqrt(((im - np.median(border, 0)) ** 2).sum(-1))
    lab, n = ndimage.label(ndimage.binary_closing(ndimage.binary_opening(diff > 0.2), np.ones((3, 3))))
    for k in range(1, n + 1):
        m = measure(lab, k)
        if m is None or m["L_px"] < 30: continue
        t = min(sc_["truth"], key=lambda t: (t["cx"] - m["cx"]) ** 2 + (t["cy"] - m["cy"]) ** 2)
        L_cm = m["L_px"] / SCALE
        K_true = 100 * a * (4.0 / t["aspect"]) ** 1.5            # synthetic truth: deeper-bodied fish are heavier
        W_from_length = a * L_cm ** b_
        rows.append(dict(L_cm=L_cm, K_true=K_true, K_from_length=100 * W_from_length / L_cm ** 3, depth_ratio=m["depth_px"] / m["L_px"]))
df = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].scatter(df.K_true, df.K_from_length, s=12); ax[0].set_xlabel("true K"); ax[0].set_ylabel("K with mass from length"); ax[0].set_title("Circular: every fish gets 1.20")
ax[1].scatter(df.K_true, df.depth_ratio, s=12); ax[1].set_xlabel("true K"); ax[1].set_ylabel("measured body depth / length"); ax[1].set_title("An image-based proxy does move")
plt.tight_layout(); plt.show()
print("spread of K computed from length:", df.K_from_length.round(6).unique())
print("correlation of depth/length ratio with true K:", round(np.corrcoef(df.K_true, df.depth_ratio)[0, 1], 2))
print("(Near-perfect only because this synthetic world was built that way. On real fish, measure the relationship")
print(" against weighed fish before trusting it.)")'''),
M("""**Project link — Capstone B.** The Mentee 2 notebook's Task 6 computes `mass_from_length(L)` and then
`condition_factor(mass, L)`, so every fish gets $K = 100 \\times 0.012 = 1.20$, exactly the lower edge of its
placeholder band, and the K flag can't respond to real condition. Fixes to discuss with your mentor:

1. weigh a sample of fish to fit $a$ and $b$, and compute $K$ only where mass is measured;
2. use an image-based shape proxy (body depth ÷ length, measured on the body, not the tail fin), **validated
   against weighed fish** before it's trusted;
3. until then, drop the K flag and say so, rather than report a number that can't change.

## 9. Biomass with an uncertainty range ▶

Report totals with a range. A bootstrap resamples the measured fish many times; we also add the calibration
uncertainty (here ±3% length bias) because it shifts every fish together."""),
C('''lengths = df.L_cm.values
rng = np.random.default_rng(1); totals = []
for _ in range(2000):
    sample = rng.choice(lengths, len(lengths), replace=True) * (1 + rng.normal(0, 0.03))
    totals.append((a * sample ** b_).sum() / 1000)
lo, hi = np.percentile(totals, [2.5, 97.5])
print(f"{len(lengths)} fish measured. Estimated biomass {np.mean(totals):.2f} kg (95% interval {lo:.2f}-{hi:.2f} kg)")'''),
M("""### 🧩 Exercise — how precise must length be?

A welfare protocol needs mass within **10%**. With $b = 3.1$, what is the largest relative length error allowed?
Store it as a fraction (e.g. `0.05` for 5%) in `max_length_error`. Use the exact relation $(1+\\varepsilon)^b = 1.10$."""),
C('''max_length_error = None   # YOUR CODE

def check_len(v):
    exp = 1.10 ** (1 / 3.1) - 1
    if v is None: print("Not yet: replace None."); return
    print(f"✅ Correct: {100 * exp:.2f}% length error" if abs(v - exp) < 2e-4 else f"❌ Expected about {exp:.4f}.")
check_len(max_length_error)'''),
C('''#@title Solution (double-click to show)
max_length_error = 1.10 ** (1 / 3.1) - 1
check_len(max_length_error)''', hidden=True),
M("""## Check yourself

<details><summary>1. Two boxes overlap by half their width and have equal size. What is their IoU?</summary>1/3: intersection is half a box; union is one and a half boxes.</details>
<details><summary>2. Why does the PR curve in turbid water stop at a low recall?</summary>Many fish never produce a detection at all, so no threshold can find them. That ceiling is a detection-probability problem.</details>
<details><summary>3. You calibrated a camera in air and use it behind a flat port. Are fish lengths too big or too small?</summary>Too big, by about a third at the centre and more toward the edges.</details>
<details><summary>4. Why is K computed from length-predicted mass useless as a welfare flag?</summary>It equals 100·a·L^(b−3); with b = 3 it's the same for every fish, so it carries no information about condition.</details>

**Next:** Module 08, *Tracking over time* — following individual fish across frames with a Kalman filter."""),
]
build(cells, "out/CV07_detection_measurement_calibration.ipynb")
