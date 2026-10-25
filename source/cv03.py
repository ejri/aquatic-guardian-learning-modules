from nbcommon import *
cells = [
M("# Module 03 — Classical image processing\n\n" + HEADER.format(num="03", time="about 2.5 hours",
  skip="you can implement convolution, Otsu thresholding and connected components, and measure a blob's length with PCA.") + """
### By the end you can
1. Implement a filter (convolution) from scratch and choose kernels for blurring, sharpening and edges.
2. Pick a threshold automatically with Otsu's method.
3. Estimate a background (from the frame border, or from many video frames) and subtract it.
4. Clean a mask with morphology and split it into objects with connected components.
5. Measure an object's area, position, orientation and length with principal component analysis (PCA).

This is the toolkit behind the **Mentee 2 segmentation baseline** (Tasks 5–6) and Module 01's pipeline.
"""),
C(SETUP, hidden=True),
C('''from scipy import ndimage
scene = make_scene(n_fish=5, seed=11)
img = underwater(scene["J"], scene["dist"], "coastal", photons=600, seed=0)
gray = img @ np.array([0.299, 0.587, 0.114])
show([img, gray], ["camera frame", "grayscale"], cmap="gray")'''),
M(r"""## 1. Filters: a small window slid across the image ▶

A filter replaces each pixel with a weighted sum of its neighbours. The weights form a **kernel**. Below is the
whole algorithm in a few lines: shift the padded image under each kernel position and add.

Strictly, this is *correlation*. True convolution flips the kernel first. For symmetric kernels they are
identical, and deep-learning libraries also compute correlation while calling it convolution."""),
C('''def conv2d(image, kernel):
    kh, kw = kernel.shape; ph, pw = kh // 2, kw // 2
    P = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(image, dtype=float)
    for dy in range(kh):
        for dx in range(kw):
            out += kernel[dy, dx] * P[dy:dy + image.shape[0], dx:dx + image.shape[1]]
    return out

KERNELS = {
    "box blur": np.ones((5, 5)) / 25,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "Sobel x (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "Sobel y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
}
outs = {k: conv2d(gray, v) for k, v in KERNELS.items()}
show([np.clip(outs["box blur"], 0, 1), np.clip(outs["sharpen"], 0, 1),
      np.abs(outs["Sobel x (vertical edges)"]), np.abs(outs["Sobel y (horizontal edges)"])],
     list(KERNELS), size=2.9, cmap="gray")
same = np.allclose(conv2d(gray, KERNELS["sharpen"]), ndimage.correlate(gray, KERNELS["sharpen"], mode="nearest"))
print("matches scipy.ndimage.correlate:", same)'''),
M(r"""**Edges** are where brightness changes fast. Combine the two Sobel responses into a gradient magnitude
$\sqrt{G_x^2 + G_y^2}$. Fish outlines light up; so does noise, which is why we usually blur first."""),
C('''gx, gy = conv2d(gray, KERNELS["Sobel x (vertical edges)"]), conv2d(gray, KERNELS["Sobel y (horizontal edges)"])
smooth = conv2d(gray, np.ones((3, 3)) / 9)
gx2, gy2 = conv2d(smooth, KERNELS["Sobel x (vertical edges)"]), conv2d(smooth, KERNELS["Sobel y (horizontal edges)"])
show([np.hypot(gx, gy), np.hypot(gx2, gy2)], ["edges, raw", "edges after a 3x3 blur"], cmap="gray")'''),
M("""✏️ **Predict, then change.** Replace the 3×3 blur with a 9×9 blur. What happens to the noise, and to the
thin tail fins? There's always a trade-off between suppressing noise and keeping detail.

## 2. From colour to a single "unusual-ness" map ▶

Fish are easier to separate by colour difference from the background than by brightness alone. As in the
Mentee notebooks, we estimate the background colour from the frame border."""),
C('''border = np.concatenate([img[:6].reshape(-1, 3), img[-6:].reshape(-1, 3), img[:, :6].reshape(-1, 3), img[:, -6:].reshape(-1, 3)])
bg_colour = np.median(border, axis=0)
diff = np.sqrt(((img - bg_colour) ** 2).sum(-1))
plt.figure(figsize=(5, 2.6)); plt.hist(diff.ravel(), bins=80); plt.yscale("log")
plt.xlabel("difference from background colour"); plt.ylabel("pixels (log)"); plt.title("Two populations: background and fish"); plt.show()'''),
M(r"""## 3. Choosing the threshold automatically: Otsu's method ▶

Otsu tries every threshold and keeps the one that best separates the histogram into two groups (it maximises the
*between-class variance* $w_0 w_1 (\mu_0 - \mu_1)^2$). Here is the full method."""),
C('''def otsu(values, bins=256):
    lo, hi = values.min(), values.max()
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    p = hist / hist.sum(); centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); mu = np.cumsum(p * centres); muT = mu[-1]
    between = (muT * w0 - mu) ** 2 / (w0 * (1 - w0) + 1e-12)
    return centres[np.argmax(between)]

from skimage.filters import threshold_otsu
thr = otsu(diff)
print(f"our Otsu threshold: {thr:.3f}   scikit-image: {threshold_otsu(diff):.3f}")
mask = diff > thr
show([diff, mask], ["difference map", f"mask (diff > {thr:.2f})"], cmap="gray")'''),
M("""## 4. Cleaning the mask: morphology ▶

- **Opening** (erode, then dilate) deletes specks smaller than the structuring element.
- **Closing** (dilate, then erode) fills small holes, like the white stripe on our fish."""),
C('''opened = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
cleaned = ndimage.binary_closing(opened, structure=np.ones((3, 3)))
show([mask, opened, cleaned], ["raw mask", "after opening", "after opening + closing"], cmap="gray")'''),
M("""## 5. Connected components: one label per object ▶

Pixels that touch belong to the same object. Below is a plain flood-fill, the same algorithm as the Mentee 2
notebook's `label_cc`, followed by SciPy's version, which runs in compiled code."""),
C('''def label_cc(m):
    H, W = m.shape; lab = np.zeros((H, W), int); cur = 0
    for i in range(H):
        for j in range(W):
            if m[i, j] and lab[i, j] == 0:
                cur += 1; stack = [(i, j)]; lab[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur; stack.append((ny, nx))
    return lab, cur

import time
t0 = time.time(); lab_py, n_py = label_cc(cleaned); t_py = time.time() - t0
t0 = time.time(); lab_sp, n_sp = ndimage.label(cleaned); t_sp = time.time() - t0
print(f"flood fill: {n_py} objects in {t_py*1000:.0f} ms   scipy: {n_sp} objects in {t_sp*1000:.1f} ms")
colours = np.random.default_rng(1).random((n_sp + 1, 3)); colours[0] = 0
show([img, colours[lab_sp]], ["frame", f"{n_sp} labelled components"])'''),
M(r"""## 6. Measuring shape with PCA ▶

Treat each object's pixel coordinates as a cloud of points. The covariance of that cloud has two eigenvectors:
the **major axis** (direction of greatest spread, along the fish) and the **minor axis** (across it). Projecting
pixels onto the major axis and taking the extent gives the length in pixels. This is exactly `measure()` in the
Mentee 2 notebook."""),
C('''def measure(lab, k):
    ys, xs = np.where(lab == k)
    if len(xs) < 20: return None
    X = np.stack([xs - xs.mean(), ys - ys.mean()])
    evals, evecs = np.linalg.eigh(np.cov(X))
    major, minor = evecs[:, np.argmax(evals)], evecs[:, np.argmin(evals)]
    along, across = X.T @ major, X.T @ minor
    L, Wd = along.max() - along.min(), across.max() - across.min()
    return dict(area=len(xs), cx=xs.mean(), cy=ys.mean(), length_px=L, width_px=Wd, aspect=L / max(Wd, 1e-6),
                angle_deg=np.degrees(np.arctan2(major[1], major[0])))

import pandas as pd
rows = []
for k in range(1, n_sp + 1):
    m = measure(lab_sp, k)
    if m is None: continue
    j = int(np.argmin([(t["cx"] - m["cx"]) ** 2 + (t["cy"] - m["cy"]) ** 2 for t in scene["truth"]]))
    m["true_length_px"] = scene["truth"][j]["length_px"]; rows.append(m)
table = pd.DataFrame(rows).round(1)
table["length_error_%"] = (100 * (table.length_px - table.true_length_px) / table.true_length_px).round(1)
table[["cx", "cy", "area", "length_px", "true_length_px", "length_error_%", "aspect", "angle_deg"]]'''),
M("""✏️ **Interpret:** is the length error mostly positive or negative? Think about what happens to the thin tail
and to edge pixels whose colour is blended with the water. A consistent bias here becomes a consistent bias in
biomass (Module 07 shows it gets roughly **tripled**).

## 7. The known failure: touching fish ▶

Connected components can't separate fish that touch. One fix: the **distance transform** (each foreground pixel's
distance to the nearest background pixel) has one peak per body; use those peaks as seeds for a **watershed**."""),
C('''from skimage.segmentation import watershed
from skimage.feature import peak_local_max
crowd = make_scene(n_fish=6, seed=8, touching=True)
cimg = underwater(crowd["J"], crowd["dist"], "coastal")
cdiff = np.sqrt(((cimg - np.median(cimg[:5].reshape(-1, 3), 0)) ** 2).sum(-1))
cmask = ndimage.binary_closing(ndimage.binary_opening(cdiff > otsu(cdiff)), np.ones((3, 3)))
lab_cc, n_cc = ndimage.label(cmask)
dist_map = ndimage.gaussian_filter(ndimage.distance_transform_edt(cmask), 2)   # smoothed: one ridge per body
peaks = peak_local_max(dist_map, min_distance=10, labels=lab_cc, exclude_border=False)
markers = np.zeros_like(lab_cc); markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
lab_ws = watershed(-dist_map, markers, mask=cmask)
pal = np.random.default_rng(3).random((max(n_cc, lab_ws.max()) + 1, 3)); pal[0] = 0
def count_big(lab, min_area=30):
    return int(sum((lab == k).sum() >= min_area for k in range(1, lab.max() + 1)))
show([cimg, pal[lab_cc], pal[lab_ws]], ["6 fish, some touching",
     f"connected components: {count_big(lab_cc)} objects", f"watershed: {count_big(lab_ws)} objects"])
print("Counts include only objects of at least 30 pixels, so specks are ignored.")
print("Look closely at the frame: is every fish in the mask at all? What happened to the darkest one?")'''),
M("""✏️ Change `min_distance` to 4 and re-run. Watershed often helps, but with loose settings it over-splits a single fish. This is the honest limit of classical
segmentation and the reason the Mentee 2 notebook's Task 10 upgrades to learned instance segmentation.

## 8. Video: a better background from many frames ▶

With a fixed camera, the **median of many frames** at each pixel is the background, because each fish occupies a
given pixel only briefly. This is how many fish-counting systems start."""),
C('''frames = [underwater(make_scene(n_fish=4, seed=100 + i, bg_seed=7)["J"], np.full((120, 180), 4.0), "coastal") for i in range(15)]
median_bg = np.median(np.stack(frames), axis=0)
fg = np.sqrt(((frames[0] - median_bg) ** 2).sum(-1))
show([frames[0], median_bg, fg], ["frame 0 of 15", "per-pixel median background", "frame 0 minus background"], cmap="gray")'''),
M("""### 🧩 Exercise — a fish-shape filter

Fish are elongated; crabs and rocks are not. Using `table` from section 6, write a boolean Series
`elongated` that is `True` where `aspect` is at least 2.5. Then count how many objects pass."""),
C('''elongated = None   # YOUR CODE, e.g. table["aspect"] >= ...

def check_elong(e):
    if e is None: print("Not yet: replace None."); return
    exp = table["aspect"] >= 2.5
    if hasattr(e, "equals") and e.equals(exp): print(f"✅ Correct: {int(exp.sum())} of {len(table)} objects pass.")
    else: print("❌ Not quite. Compare with table['aspect'] >= 2.5.")
check_elong(elongated)'''),
C('''#@title Solution (double-click to show)
elongated = table["aspect"] >= 2.5
check_elong(elongated)''', hidden=True),
M("""## Check yourself

<details><summary>1. Why blur before computing edges?</summary>Edge kernels amplify pixel-to-pixel noise; blurring suppresses noise first, at the cost of some fine detail.</details>
<details><summary>2. What does Otsu assume about the histogram?</summary>That it's roughly two groups (background and objects). It struggles when objects are tiny, or when there are more than two populations.</details>
<details><summary>3. Two fish touch and connected components returns one object with double the area. Name two fixes.</summary>Distance transform + watershed, or a learned instance-segmentation model. Better camera siting (fewer overlaps) helps too.</details>
<details><summary>4. Why is a median background better than a mean for video?</summary>The median ignores the occasional frames where a fish covers the pixel; the mean gets pulled toward fish colours.</details>

**Next:** Module 04, *Learning to classify* — from hand-made rules to a model that learns its own decision
boundary, and how to evaluate it without fooling yourself."""),
]
build(cells, "out/CV03_classical_image_processing.ipynb")
