from nbcommon import *
cells = [
M("# Module 02 — How water changes an image\n\n" + HEADER.format(num="02", time="about 2 hours",
  skip="you can already write down an underwater image-formation model and explain why inverting it fails at range.") + """
### By the end you can
1. Write the simple underwater image-formation model and say what each term means physically.
2. Explain why red disappears first and why turbid water looks green or brown.
3. Compute a **detection range**: the distance beyond which a fish stops being distinguishable.
4. Say what the project's `degrade()` utility assumes, and where it differs from the physics.
5. Show why "restoring" a distant image amplifies noise, and why no filter can bring back what the water removed.
"""),
C(SETUP, hidden=True),
M(r"""## 1. The model ▶

Light leaving a fish travels a distance $d$ through water to the camera. Two things happen on the way:

- **Direct signal fades.** A fraction $t_c = e^{-\beta_c d}$ survives, separately for each colour channel $c$.
  $\beta_c$ is the attenuation coefficient (per metre). Water absorbs red far faster than blue or green.
- **Veiling light is added.** Particles scatter ambient light toward the camera along the same path. This
  "backscatter" approaches the water's own colour $B_c$ as the path gets long.

$$I_c \;=\; \underbrace{J_c\,t_c}_{\text{what's left of the fish}} \;+\; \underbrace{B_c\,(1-t_c)}_{\text{veiling light}}, \qquad t_c = e^{-\beta_c d}$$

$J_c$ is what the fish would look like in air; $I_c$ is what the camera records. This is a simplified version
of models used in underwater imaging research. The coefficients in `WATER` are round teaching values with the
right ordering, not site measurements."""),
C('''d = np.linspace(0, 15, 300)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for a, (name, w) in zip(ax, WATER.items()):
    for beta, col, lab in zip(w["beta"], ("#c0392b", "#2e8b57", "#2f5fae"), "RGB"):
        a.plot(d, np.exp(-beta * d), color=col, lw=2, label=f"{lab}: beta={beta}/m")
    a.set_title(name); a.set_xlabel("distance (m)"); a.legend(fontsize=8)
ax[0].set_ylabel("fraction of light surviving, t")
plt.tight_layout(); plt.show()'''),
M("""✏️ **Predict:** in `coastal` water, at what distance has half the red light gone? Work it out on paper
($t = 0.5 \\Rightarrow d = \\ln 2 / \\beta$), then check below."""),
C('''for name, w in WATER.items():
    half = np.log(2) / np.array(w["beta"])
    print(f"{name:15s} half-distance  R {half[0]:5.1f} m   G {half[1]:5.1f} m   B {half[2]:5.1f} m")'''),
M("""## 2. The same fish, farther away ▶

We place every fish at the same distance and move them back."""),
C('''scene = make_scene(n_fish=4, seed=7)
fish_pixels = np.any(scene["masks"], axis=0)
def at_distance(scene, fish_m, bg_m=None):
    dist = np.full(fish_pixels.shape, float(bg_m if bg_m is not None else fish_m + 6.0))
    dist[fish_pixels] = fish_m
    return dist

for water in WATER:
    frames = [underwater(scene["J"], at_distance(scene, dm), water) for dm in (1, 3, 6, 10)]
    show(frames, [f"{water}, {dm} m" for dm in (1, 3, 6, 10)], size=2.8)'''),
M("""## 3. Contrast and detection range ▶

A detector can only find a fish if the fish looks different from what's behind it. Define contrast as the
colour distance between fish pixels and background pixels. As $t \\to 0$, both approach the veiling light $B$,
so contrast falls toward zero **whatever the fish looks like**.

We will call a fish *detectable* if its contrast exceeds 0.08, the same threshold as Module 01's pipeline.
That number is a stand-in for any real detector's sensitivity."""),
C('''THRESH = 0.08
dists = np.linspace(0.5, 20, 80)
def contrast_at(water, dm):
    I = underwater(scene["J"], at_distance(scene, dm), water)
    fish_mean = I[fish_pixels].mean(0); bg_mean = I[~fish_pixels].mean(0)
    return np.sqrt(((fish_mean - bg_mean) ** 2).sum())

detection_range = {}
fig, ax = plt.subplots(figsize=(7, 3.4))
for water in WATER:
    c = np.array([contrast_at(water, dm) for dm in dists])
    above = dists[c > THRESH]
    detection_range[water] = above.max() if len(above) else 0.0
    ax.plot(dists, c, lw=2, label=f"{water} (range ~ {detection_range[water]:.1f} m)")
ax.axhline(THRESH, color="k", ls="--", lw=1); ax.text(19.5, THRESH + 0.01, "detector threshold", ha="right", fontsize=8)
ax.set_xlabel("fish distance (m)"); ax.set_ylabel("fish-background contrast"); ax.legend(fontsize=8)
ax.set_title("Contrast falls with distance; turbid water falls faster"); plt.tight_layout(); plt.show()'''),
M("""**Why this matters for the project.** If turbidity rises near an outfall, the camera's detection range
shrinks. Fewer fish are *seen* even if exactly as many are *there*. Module 09 turns this into a statistical
correction (detection probability and occupancy)."""),
M("""## 4. Comparing with the project's `degrade()` utility ▶

The Aquatic Guardian notebooks test robustness with `degrade(img, turbidity, low_light, blur_k)`. It blends the
image toward a fixed green-blue veil and partly toward its own mean. Read the code in the setup cell, then compare.

Its assumptions, written out:
- the same transmittance for **every** channel (no wavelength dependence);
- the same transmittance for **every** pixel (no dependence on distance);
- a fixed veil colour regardless of water type.

These are reasonable for a quick robustness test. They are not a physical model. Let's measure the gap."""),
C('''target = underwater(scene["J"], at_distance(scene, 6), "coastal")      # physics at 6 m
grid = np.linspace(0, 0.95, 96)
errs = [np.abs(degrade(scene["J"], turbidity=tb) - target).mean() for tb in grid]
best = grid[int(np.argmin(errs))]
approx = degrade(scene["J"], turbidity=best)
per_channel = np.abs(approx - target).mean((0, 1))
show([target, approx, np.abs(approx - target).sum(-1)],
     ["physics: coastal, 6 m", f"degrade(turbidity={best:.2f}), best fit", "absolute difference"], cmap="magma")
print("mean abs error per channel (R, G, B):", per_channel.round(3))'''),
M("""✏️ **Interpret:** which channel does `degrade()` get most wrong, and why? (Look at its veil colour and at
which channel the physics removes fastest.) When you report robustness results in Capstone A, say which
degradation model you used."""),
M(r"""## 5. Can we undo it? Restoration and its limits ▶

If we knew $\beta$, $B$ and $d$ exactly, we could invert the model:

$$\hat J_c = \frac{I_c - B_c(1-t_c)}{t_c}.$$

Dividing by $t_c$ is the catch. Any error in $I_c$ — sensor noise, or simply storing the image as 8-bit
whole numbers — is multiplied by $1/t_c$. At long range $t_c$ is tiny, so the error explodes."""),
C('''def restore(I, dist, water):
    beta = np.array(WATER[water]["beta"]); B = np.array(WATER[water]["veil"])
    t = np.exp(-beta[None, None, :] * dist[..., None])
    return (I - B * (1 - t)) / t

rows = []
for dm in (1, 3, 6, 10, 15, 20):
    dist = at_distance(scene, dm, bg_m=dm)
    I = underwater(scene["J"], dist, "turbid_harbour")
    exact = restore(I, dist, "turbid_harbour")
    stored8 = restore(np.round(I * 255) / 255, dist, "turbid_harbour")          # what a camera file keeps
    noisy = restore(np.clip(I + np.random.default_rng(0).normal(0, 0.005, I.shape), 0, 1), dist, "turbid_harbour")
    t_red = np.exp(-WATER["turbid_harbour"]["beta"][0] * dm)
    rows.append((dm, t_red, np.abs(exact - scene["J"]).mean(), np.abs(stored8 - scene["J"]).mean(),
                 np.abs(noisy - scene["J"]).mean()))
print(f"{'dist':>5} {'t_red':>10} {'err exact':>10} {'err 8-bit':>10} {'err noise':>10}")
for r in rows: print(f"{r[0]:5.0f} {r[1]:10.2e} {r[2]:10.2e} {r[3]:10.3f} {r[4]:10.3f}")'''),
M("""Reading the table: with perfect numbers the inversion is essentially exact (errors far too small to see). Once the image is stored as
8-bit values, or has a little sensor noise, the restoration error grows roughly like $1/t$ and the
"restored" image is mostly amplified noise.

**Project link.** The Aquatic Guardian team hit exactly this: a round-trip test of the image-formation model
failed at long range in turbid water because transmittance near $5\\times10^{-8}$ amplified rounding by about
20 million times. The fix was to report a conditioning table, not to loosen the test tolerance. The lesson for
you: **when a check fails, find out why before changing the check.**"""),
C('''dm = 10
dist = at_distance(scene, dm, bg_m=dm)
I = underwater(scene["J"], dist, "turbid_harbour")
stored8 = np.clip(restore(np.round(I * 255) / 255, dist, "turbid_harbour"), 0, 1)
show([I, stored8, scene["J"]], [f"camera, {dm} m harbour", "restored from 8-bit file", "truth (in air)"])'''),
M("""## 6. Enhancement: looks better, knows nothing new ▶

Common fixes: **gray-world white balance** (scale channels so their means match), **contrast stretching**, and
**CLAHE** (adaptive histogram equalisation). They can make a frame easier for a person or a fixed-threshold
detector to read. They cannot recover detail that fell below the noise. Compare what each does to the noise."""),
C('''from skimage import exposure
I = underwater(scene["J"], at_distance(scene, 6), "coastal", photons=300, seed=1)

def gray_world(img):
    m = img.mean((0, 1)); return np.clip(img * (m.mean() / (m + 1e-6)), 0, 1)
def stretch(img, lo=1, hi=99):
    out = np.empty_like(img)
    for c in range(3):
        a, b = np.percentile(img[..., c], (lo, hi)); out[..., c] = np.clip((img[..., c] - a) / (b - a + 1e-6), 0, 1)
    return out
clahe = exposure.equalize_adapthist(I, clip_limit=0.02)
versions = {"camera": I, "gray-world": gray_world(I), "stretch": stretch(I), "CLAHE": clahe}
show(list(versions.values()), list(versions.keys()), size=2.9)
bgpatch = (slice(2, 14), slice(2, 30))      # a fish-free corner
for k, v in versions.items():
    print(f"{k:11s} background noise (per-channel std in a flat corner, averaged): {v[bgpatch].std(axis=(0, 1)).mean():.4f}")'''),
M("""Every enhancement that boosts contrast also boosts noise in flat regions. Enhancement changes how
information is *displayed*; it does not create information.

## 7. Low light ▶

Fewer photons means more relative noise (shot noise grows like $\\sqrt{N}$ while signal grows like $N$)."""),
C('''frames = [underwater(scene["J"], at_distance(scene, 3), "coastal", light=L, photons=400, seed=2) for L in (1.0, 0.3, 0.08)]
show([f / f.max() for f in frames], ["full light", "dusk (0.3)", "night-ish (0.08), rescaled"], size=3)'''),
M("""### 🧩 Exercise — detection range for a big, dark fish

Detection range depends on the target too. Recompute `detection_range` for `coastal` water using a scene where
every fish is dark grey `(0.15, 0.15, 0.15)` instead of colourful. Store a single number (metres) in
`dark_range`. Before you run it, predict: longer or shorter than the colourful fish?"""),
C('''dark_range = None   # YOUR CODE (hint: copy scene["J"], set fish pixels to 0.15, reuse the contrast idea)

def check_dark(r):
    J = scene["J"].copy(); J[fish_pixels] = 0.15
    def c(dm):
        I = underwater(J, at_distance(scene, dm), "coastal")
        return np.sqrt(((I[fish_pixels].mean(0) - I[~fish_pixels].mean(0)) ** 2).sum())
    cs = np.array([c(dm) for dm in dists]); above = dists[cs > THRESH]
    expected = above.max() if len(above) else 0.0
    if r is None: print("Not yet: replace None with a number in metres."); return
    if abs(r - expected) < 0.6: print(f"✅ About right: {expected:.1f} m (colourful fish: {detection_range['coastal']:.1f} m).")
    else: print(f"❌ Expected about {expected:.1f} m, got {r}.")
check_dark(dark_range)'''),
C('''#@title Solution (double-click to show)
J_dark = scene["J"].copy(); J_dark[fish_pixels] = 0.15
cs = []
for dm in dists:
    I = underwater(J_dark, at_distance(scene, dm), "coastal")
    cs.append(np.sqrt(((I[fish_pixels].mean(0) - I[~fish_pixels].mean(0)) ** 2).sum()))
cs = np.array(cs); dark_range = dists[cs > THRESH].max() if (cs > THRESH).any() else 0.0
check_dark(dark_range)''', hidden=True),
M("""## Check yourself

<details><summary>1. In the model, what happens to I as distance grows very large?</summary>t → 0, so I → B: every pixel becomes the water's veiling colour and all contrast disappears.</details>
<details><summary>2. Why does turbid coastal water often look green rather than blue?</summary>Dissolved organic matter and particles absorb blue strongly, and water absorbs red; green survives best.</details>
<details><summary>3. A colleague says "we used CLAHE, so turbidity no longer affects our counts." What do you ask?</summary>Did they measure detection rate by turbidity after enhancement? Enhancement can't restore signal below the noise, so detection still falls with range and turbidity.</details>
<details><summary>4. Name one assumption of the project's degrade() that the physics model doesn't make.</summary>Same transmittance for every channel and every pixel, or a fixed veil colour.</details>

**Next:** Module 03, *Classical image processing* — filters, thresholds and shapes, the tools behind the
Module 01 pipeline and the Mentee 2 segmentation baseline."""),
]
build(cells, "out/CV02_how_water_changes_images.ipynb")
