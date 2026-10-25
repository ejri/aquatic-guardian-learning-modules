from nbcommon import *
cells = [
M("# Module 01 — Images are arrays of numbers\n\n" + HEADER.format(num="01", time="about 1.5 hours",
  skip="you already index NumPy arrays comfortably and know what shape `(H, W, 3)` means.") + """
### By the end you can
1. Explain what a digital image is in terms a computer can work with.
2. Index, crop and draw on an image with NumPy slicing, using `img[y, x]` correctly.
3. Split an image into colour channels and read a histogram.
4. Avoid the two classic bugs: swapped `x`/`y`, and `uint8` overflow.
"""),
C(SETUP, hidden=True),
M("""## 1. The whole game first ▶

Before any detail, here is what you will be able to build by the end of the track. The cell below makes a
synthetic underwater frame with five fish, measures how different each pixel is from the background,
groups the different-looking pixels into blobs, and draws a box around each blob.

You will not follow every line yet. That's fine: by Module 07 you can write each one yourself."""),
C('''from scipy import ndimage

WATER_TYPE = "coastal"          # ✏️ later: try "clear_ocean" and "turbid_harbour"

scene = make_scene(n_fish=5, seed=3)
img = underwater(scene["J"], scene["dist"], WATER_TYPE)

border = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3),
                         img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
background = np.median(border, axis=0)                       # typical background colour
difference = np.sqrt(((img - background) ** 2).sum(axis=-1))  # how unusual each pixel looks
mask = ndimage.binary_opening(difference > 0.08)              # keep clearly unusual pixels
labels, n = ndimage.label(mask)                               # group touching pixels into blobs
boxes = []
for sl in ndimage.find_objects(labels):
    if (labels[sl] > 0).sum() > 30:                           # ignore tiny specks
        boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))

show([img, difference, draw_boxes(img, boxes)],
     ["camera frame (" + WATER_TYPE + ")", "difference from background", f"{len(boxes)} boxes"], cmap="gray")
print(f"Fish actually in the scene: {len(scene['boxes'])}.  Boxes found: {len(boxes)}.")'''),
M("""✏️ **Predict, then change.** Set `WATER_TYPE = "turbid_harbour"` and re-run. Before you do, write down how
many boxes you expect.

Then ask the question this whole project turns on: **did the fish leave, or did the camera stop seeing them?**
Module 02 explains the physics. Module 09 explains why confusing the two can produce a false environmental finding.

---
## 1b. The same idea on a real animal ▶

Synthetic fish let us know the truth. The cell below loads a **real** wildlife photograph (a green sea turtle
photographed by NOAA, public domain) and a live-camera still of a reef school, and runs the same
difference-from-background detector. Classification here is just a label we give each frame: turtle, school,
shark, manatee. Detection is the boxes."""),
C('''real = {
    "Green turtle (NOAA)": load_photo("noaa-honu"),
    "Reef school": load_photo("reef-school"),
    "Nurse shark": load_photo("nurse-shark"),
    "Manatees": load_photo("manatees"),
}
real_boxes = {}
for title, frame in real.items():
    border = np.concatenate([frame[:6].reshape(-1, 3), frame[-6:].reshape(-1, 3),
                             frame[:, :6].reshape(-1, 3), frame[:, -6:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    difference = np.sqrt(((frame - background) ** 2).sum(axis=-1))
    mask = ndimage.binary_opening(difference > 0.12)
    labels, _ = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        if sl and (labels[sl] > 0).sum() > 80:
            boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    real_boxes[title] = boxes
    show([frame, draw_boxes(frame, boxes)],
         [title + "  ·  class: " + title.split()[0].lower(), f"{len(boxes)} detection box(es)"])
print("Frames classified (one label per photograph):", list(real))
print("Boxes found:", {k: len(v) for k, v in real_boxes.items()})'''),
M("""Those boxes are not ground truth. A person still has to say whether a box is a turtle, a rock, or a
diver. The rest of this module treats the image as numbers so you can see why that check is needed.

---
## 2. An image is a grid of numbers ▶

A colour image is a 3-D array: **rows × columns × channels**. Row 0 is the top. Each pixel holds three numbers:
how much red, green and blue light reached that sensor element. In these notebooks values are floats from
0 (none) to 1 (full). Image files and OpenCV usually store whole numbers 0–255 (`uint8`)."""),
C('''tiny = np.zeros((4, 6, 3))          # 4 rows (height), 6 columns (width), 3 channels
tiny[1, 2] = [1.0, 0.5, 0.0]        # row 1, column 2 -> orange
tiny[3, 5] = [0.0, 0.3, 0.9]        # bottom-right -> blue
print("shape:", tiny.shape, " -> (height, width, channels)")
print("pixel at row 1, column 2:", tiny[1, 2])
plt.figure(figsize=(3, 2)); plt.imshow(tiny, interpolation="nearest")
plt.title("a 4 x 6 image, zoomed in"); plt.xticks(range(6)); plt.yticks(range(4)); plt.show()'''),
M("""**The `x`/`y` trap.** Maths and plotting use `(x, y)` = (across, down). Arrays use `[row, column]` =
`[y, x]`. So the pixel at x=5, y=3 is `img[3, 5]`. Boxes in this track are stored as `(x1, y1, x2, y2)`,
so cropping a box is `img[y1:y2, x1:x2]`. Getting this backwards is the most common bug in student CV code."""),
C('''print("our camera frame:", img.shape, img.dtype, "min", img.min().round(3), "max", img.max().round(3))
t = scene["truth"][0]
x, y = int(t["cx"]), int(t["cy"])
print(f"fish 0 centre at x={x}, y={y}: pixel value img[{y}, {x}] =", img[y, x].round(3))
print("a background pixel img[5, 5] =", img[5, 5].round(3))'''),
M("""## 3. Three channels ▶

Look at each channel on its own. Brighter means more of that colour reached the camera."""),
C('''R, G, B = img[..., 0], img[..., 1], img[..., 2]
show([R, G, B], ["red channel", "green channel", "blue channel"], cmap="gray")
fig, ax = plt.subplots(figsize=(6, 2.6))
for ch, name, col in zip((R, G, B), ("red", "green", "blue"), ("#c0392b", "#2e8b57", "#2f5fae")):
    ax.hist(ch.ravel(), bins=50, range=(0, 1), histtype="step", lw=1.6, color=col, label=name)
ax.set_xlabel("pixel value"); ax.set_ylabel("number of pixels"); ax.legend(); ax.set_title("channel histograms")
plt.tight_layout(); plt.show()'''),
M("""✏️ **Predict:** which channel has the lowest average value in this underwater frame, and which has the
lowest in the *in-air* scene `scene["J"]`? Write both down, then run the next cell."""),
C('''print("underwater means (R, G, B):", img.mean(axis=(0, 1)).round(3))
print("in-air means     (R, G, B):", scene["J"].mean(axis=(0, 1)).round(3))
print("Red is absorbed fastest by water. Module 02 turns this into an equation you can invert.")'''),
M("""## 4. Cropping and drawing with slicing ▶"""),
C('''x1, y1, x2, y2 = scene["boxes"][0]
crop = img[y1:y2, x1:x2]                       # rows first, then columns
marked = img.copy()
marked[y1, x1:x2] = (1, 0.85, 0); marked[y2 - 1, x1:x2] = (1, 0.85, 0)   # top and bottom edges
marked[y1:y2, x1] = (1, 0.85, 0); marked[y1:y2, x2 - 1] = (1, 0.85, 0)   # left and right edges
show([marked, crop], ["box drawn with slicing", f"crop, shape {crop.shape}"])'''),
M("""### 🧩 Exercise 1 — crop the largest fish

`scene["boxes"]` holds one `(x1, y1, x2, y2)` box per fish. Find the box with the largest area
(`(x2 - x1) * (y2 - y1)`) and crop it from `img`. Replace `None`."""),
C('''largest_crop = None   # YOUR CODE

def check_crop(c):
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in scene["boxes"]]
    x1, y1, x2, y2 = scene["boxes"][int(np.argmax(areas))]
    if c is None: print("Not yet: replace None with your crop."); return
    if c.shape == (y2 - y1, x2 - x1, 3) and np.allclose(c, img[y1:y2, x1:x2]): print("✅ Correct.")
    elif c.shape == (x2 - x1, y2 - y1, 3): print("❌ Shape is transposed. Did you slice [x, y] instead of [y, x]?")
    else: print("❌ Not the largest box yet. Expected shape", (y2 - y1, x2 - x1, 3), "got", c.shape)
check_crop(largest_crop)'''),
C('''#@title Solution (double-click to show)
areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in scene["boxes"]]
x1, y1, x2, y2 = scene["boxes"][int(np.argmax(areas))]
largest_crop = img[y1:y2, x1:x2]
check_crop(largest_crop)''', hidden=True),
M("""## 5. Data types bite: `uint8` overflow ▶

Images loaded from files are usually `uint8` (0–255). Arithmetic on them wraps around instead of saturating."""),
C('''a = np.array([200], dtype=np.uint8)
print("200 + 100 as uint8   =", a + np.uint8(100), " <- wrapped around past 255")
print("200 + 100 as float   =", a.astype(float) + 100)
img8 = (img * 255).astype(np.uint8)
brighter_bad = img8 + np.uint8(120)                                   # wraps: bright pixels turn dark
brighter_ok = np.clip(img8.astype(np.int16) + 120, 0, 255).astype(np.uint8)
show([img8, brighter_bad, brighter_ok], ["uint8 frame", "+120, wrapped (bug)", "+120, clipped (correct)"])
print("Rule: convert to float (img8 / 255.0) before doing maths, convert back at the end.")'''),
M("""## 6. Grayscale and hue ▶

Many classical methods work on one channel. A standard grayscale weights green most because our eyes are
most sensitive to it. **Hue** describes colour independent of brightness, which can help when water dims a scene."""),
C('''from matplotlib.colors import rgb_to_hsv
gray = 0.299 * R + 0.587 * G + 0.114 * B
hsv = rgb_to_hsv(img)
show([gray, hsv[..., 0], hsv[..., 1]], ["grayscale", "hue", "saturation"], cmap="gray")'''),
M("""### 🧩 Exercise 2 — the brightest fish pixel

Using `gray`, find the `(x, y)` location of the brightest pixel. Hint: `np.argmax` gives a flat index;
`np.unravel_index(i, gray.shape)` turns it into `(row, column)`. Remember which one is `y`."""),
C('''brightest_xy = None   # YOUR CODE: a tuple (x, y)

def check_brightest(xy):
    r, c = np.unravel_index(np.argmax(gray), gray.shape)
    if xy is None: print("Not yet: replace None with (x, y)."); return
    if tuple(map(int, xy)) == (c, r): print("✅ Correct: x =", c, "y =", r)
    elif tuple(map(int, xy)) == (r, c): print("❌ Swapped. unravel_index returns (row, column) = (y, x).")
    else: print("❌ Not the brightest pixel.")
check_brightest(brightest_xy)'''),
C('''#@title Solution (double-click to show)
r, c = np.unravel_index(np.argmax(gray), gray.shape)
brightest_xy = (c, r)
check_brightest(brightest_xy)''', hidden=True),
M("""## 7. Your own images (optional)

Set `USE_OWN_IMAGE = True` in Colab to upload a photo. Record where it came from and its licence before you
use it in any project work."""),
C('''USE_OWN_IMAGE = False
if USE_OWN_IMAGE:
    from google.colab import files
    import matplotlib.image as mpimg
    name = list(files.upload().keys())[0]
    own = mpimg.imread(name)
    own = own[..., :3].astype(float) / (255.0 if own.dtype == np.uint8 else 1.0)
    print(name, own.shape, own.dtype); show(own, [name])
else:
    print("Skipped (USE_OWN_IMAGE = False).")'''),
M("""## Check yourself

<details><summary>1. An image has shape (480, 640, 3). How wide is it?</summary>640 pixels. Shape is (height, width, channels).</details>
<details><summary>2. A box is (x1, y1, x2, y2) = (10, 20, 50, 60). Write the crop.</summary><code>img[20:60, 10:50]</code> — rows (y) first.</details>
<details><summary>3. Why might red be the darkest channel underwater?</summary>Water absorbs long (red) wavelengths fastest. Module 02 quantifies it.</details>
<details><summary>4. You add 50 to a uint8 image and the sky turns black. What happened?</summary>Values above 255 wrapped around. Convert to float or a wider integer type, then clip.</details>

**Next:** Module 02, *How water changes an image* — the physics behind the missing fish in section 1."""),
]
build(cells, "out/CV01_images_are_arrays.ipynb")
