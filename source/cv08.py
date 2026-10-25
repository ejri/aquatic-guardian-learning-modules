from nbcommon import *
TRACK_SETUP = r'''#@title ▶ Setup: a simulated school of fish seen from above
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.optimize import linear_sum_assignment
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 10})
W_PX, H_PX = 640, 360

def simulate_school(n_fish=8, T=200, speed=3.0, turn=0.12, align=0.0, seed=0):
    """Positions and velocities (pixels, pixels/frame) for n_fish over T frames.
    turn: random heading change per frame (radians). align: pull toward the group's mean heading (0 = none)."""
    r = np.random.default_rng(seed)
    pos = np.column_stack([r.uniform(60, W_PX - 60, n_fish), r.uniform(60, H_PX - 60, n_fish)])
    head = r.uniform(-np.pi, np.pi, n_fish) if align < 0.5 else r.normal(0, 0.4, n_fish)
    P, V = np.zeros((T, n_fish, 2)), np.zeros((T, n_fish, 2))
    for t in range(T):
        mean_dir = np.angle(np.exp(1j * head).mean())
        head = head + r.normal(0, turn, n_fish) + align * np.angle(np.exp(1j * (mean_dir - head)))
        v = speed * np.column_stack([np.cos(head), np.sin(head)])
        pos = pos + v
        for d, lim in ((0, W_PX), (1, H_PX)):                       # bounce off the tank walls
            out = (pos[:, d] < 10) | (pos[:, d] > lim - 10)
            v[out, d] *= -1; pos[:, d] = np.clip(pos[:, d], 10, lim - 10)
        head = np.arctan2(v[:, 1], v[:, 0]); P[t], V[t] = pos, v
    return P, V

def observe(P, p_detect=0.95, sigma=1.5, fp_per_frame=0.3, turbid_x=None, p_turbid=0.35, seed=1):
    """Noisy detections per frame: list of arrays with columns x, y, true_id (-1 = false positive).
    Inside turbid_x = (x0, x1) the detection probability drops to p_turbid."""
    r = np.random.default_rng(seed); frames = []
    for t in range(P.shape[0]):
        rows = []
        for i, (x, y) in enumerate(P[t]):
            p = p_turbid if (turbid_x and turbid_x[0] <= x <= turbid_x[1]) else p_detect
            if r.random() < p: rows.append((x + r.normal(0, sigma), y + r.normal(0, sigma), i))
        for _ in range(r.poisson(fp_per_frame)): rows.append((r.uniform(0, W_PX), r.uniform(0, H_PX), -1))
        frames.append(np.array(rows) if rows else np.zeros((0, 3)))
    return frames
print("Simulator ready.")'''
cells = [
M("# Module 08 — Tracking over time\n\n" + HEADER.format(num="08", time="about 2.5 hours",
  skip="you have implemented a Kalman filter and a SORT-style tracker, and you report ID switches and track purity.") + """
### By the end you can
1. Explain **tracking-by-detection**: detect in each frame, then link detections into tracks.
2. Implement a constant-velocity **Kalman filter** and use it to coast through missed detections.
3. Assign detections to tracks optimally with the **Hungarian algorithm**, with a gate.
4. Measure tracker quality: tracks per fish, **ID switches**, and **track purity**.
5. Work out the **camera-siting** requirement: how far a fish may move between frames before tracking breaks.
6. Estimate a school-level behaviour metric (**polarisation**) and compare velocity sources honestly.

This module prepares the tracking and behaviour parts of the monitoring and aquaculture notebooks.
"""),
C(TRACK_SETUP, hidden=True),
C('''P, V = simulate_school(n_fish=8, T=200, seed=0)
dets = observe(P, turbid_x=(260, 380), seed=1)
fig, ax = plt.subplots(figsize=(9, 5))
ax.axvspan(260, 380, color="#7a8b6f", alpha=0.25, label="turbid patch (detection p = 0.35)")
for i in range(P.shape[1]): ax.plot(P[:, i, 0], P[:, i, 1], lw=1, alpha=0.6)
D = np.vstack([d for d in dets if len(d)])
ax.scatter(D[D[:, 2] >= 0, 0], D[D[:, 2] >= 0, 1], s=3, c="k", label="detections")
ax.scatter(D[D[:, 2] < 0, 0], D[D[:, 2] < 0, 1], s=12, marker="x", c="r", label="false positives")
ax.set_xlim(0, W_PX); ax.set_ylim(H_PX, 0); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
ax.set_title("Truth (lines) and what the detector gives us (dots): no identities attached"); plt.show()'''),
M(r"""## 1. The Kalman filter, one fish ▶

The state is position and velocity, $\mathbf{s} = (x, y, v_x, v_y)$. Each frame:

1. **Predict**: move the state forward assuming constant velocity, and grow the uncertainty $P$ by the
   **process noise** $Q$ (how much the fish might change course).
2. **Update** (if there's a detection): blend prediction and measurement. The **Kalman gain** decides the weights:
   trust the measurement more when its noise $R$ is small relative to the prediction's uncertainty.

With no detection, skip the update and **coast** on the prediction. Uncertainty grows until the fish reappears."""),
C('''class KalmanCV:
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    def __init__(self, z, q=0.5, r=2.0):
        self.s = np.array([z[0], z[1], 0.0, 0.0]); self.P = np.diag([r, r, 25.0, 25.0])
        G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]]); self.Q = q * G @ G.T; self.R = r * np.eye(2)
    def predict(self):
        self.s = self.F @ self.s; self.P = self.F @ self.P @ self.F.T + self.Q; return self.s[:2].copy()
    def update(self, z):
        y = np.asarray(z) - self.Hm @ self.s; S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S); self.s = self.s + K @ y; self.P = (np.eye(4) - K @ self.Hm) @ self.P

fish = 0; truth = P[:, fish]
r = np.random.default_rng(3); meas = truth + r.normal(0, 4.0, truth.shape)
seen = np.ones(len(truth), bool); seen[70:95] = False; seen[r.random(len(truth)) < 0.15] = False
kf = KalmanCV(meas[0], q=0.3, r=16.0); est = []
for t in range(len(truth)):
    kf.predict()
    if seen[t] and t > 0: kf.update(meas[t])
    est.append(kf.s[:2].copy())
est = np.array(est)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(truth[:, 0], truth[:, 1], "k-", lw=1, label="truth")
ax.plot(meas[seen, 0], meas[seen, 1], ".", ms=4, color="#999", label="noisy detections")
ax.plot(est[:, 0], est[:, 1], "-", lw=2, color="#1f6f8b", label="Kalman estimate")
ax.plot(truth[70:95, 0], truth[70:95, 1], "-", lw=6, color="#e3b505", alpha=0.35, label="25 frames with no detection")
ax.set_ylim(H_PX, 0); ax.set_aspect("equal"); ax.legend(fontsize=8); plt.show()
err = lambda a, b: np.linalg.norm(a - b, axis=1).mean()
gap = np.zeros(len(truth), bool); gap[70:95] = True
print(f"frames with a detection:  raw detection error {err(meas[seen], truth[seen]):.2f} px   Kalman error {err(est[seen], truth[seen]):.2f} px")
print(f"the 25-frame gap:          no raw detection at all         Kalman error {err(est[gap], truth[gap]):.2f} px (coasting)")'''),
M("""✏️ **Predict, then change.** Set `q=5.0` (expect sharp turns) and re-run. Then `q=0.01`. Which follows the
wiggles better, and which coasts through the gap better? There's no free lunch: $Q$ trades responsiveness for smoothness.

## 2. Many fish: who is who? ▶

Each frame we must decide which detection belongs to which track. The **Hungarian algorithm**
(`scipy.optimize.linear_sum_assignment`) finds the assignment with the smallest total distance. A **gate** forbids
matches that are too far apart. Unmatched detections start new tentative tracks; a track is **confirmed** after
`min_hits` matches and **deleted** after `max_age` frames without one. This is the logic of SORT, using point
distance instead of box overlap."""),
C('''def run_tracker(frames, use_kalman=True, q=0.5, r=2.0, gate=25.0, min_hits=3, max_age=10):
    live, done, next_id = [], [], 0
    for t, D in enumerate(frames):
        preds = []
        for tr in live:
            preds.append(tr["kf"].predict() if use_kalman else tr["last"])
        matched_t, matched_d = set(), set()
        if live and len(D):
            cost = np.linalg.norm(np.array(preds)[:, None, :] - D[None, :, :2], axis=2)
            for i, j in zip(*linear_sum_assignment(np.where(cost > gate, 1e6, cost))):
                if cost[i, j] <= gate:
                    tr = live[i]; z = D[j, :2]
                    if use_kalman: tr["kf"].update(z)
                    tr["last"] = z; tr["hits"] += 1; tr["miss"] = 0
                    tr["hist"].append((t, z[0], z[1], int(D[j, 2]))); tr["vel"].append(tr["kf"].s[2:].copy() if use_kalman else None)
                    matched_t.add(i); matched_d.add(j)
        for i, tr in enumerate(live):
            if i not in matched_t: tr["miss"] += 1
        for j in range(len(D)):
            if j not in matched_d:
                z = D[j, :2]
                live.append(dict(id=next_id, kf=KalmanCV(z, q, r), last=z, hits=1, miss=0,
                                 hist=[(t, z[0], z[1], int(D[j, 2]))], vel=[None])); next_id += 1
        keep = []
        for tr in live: (done if tr["miss"] > max_age else keep).append(tr)
        live = keep
    return [tr for tr in done + live if tr["hits"] >= min_hits]

def score(tracks, n_fish):
    purity, owner = [], {}
    for tr in tracks:
        ids = np.array([h[3] for h in tr["hist"]]); real = ids[ids >= 0]
        purity.append(np.bincount(real).max() / len(ids) if len(real) else 0.0)
        for (t, _, _, i) in tr["hist"]:
            if i >= 0: owner.setdefault(i, []).append((t, tr["id"]))
    switches = 0
    for i, seq in owner.items():
        ids = [k for _, k in sorted(seq)]; switches += sum(a != b for a, b in zip(ids, ids[1:]))
    return dict(fish=n_fish, confirmed_tracks=len(tracks), id_switches=switches,
                mean_purity=round(float(np.mean(purity)), 3) if purity else 0.0)

res = {"nearest neighbour (no prediction)": score(run_tracker(dets, use_kalman=False), 8),
       "Kalman + Hungarian": score(run_tracker(dets, use_kalman=True), 8)}
pd.DataFrame(res).T'''),
M("""✏️ **Interpret:** the ideal is 8 confirmed tracks, 0 ID switches, purity 1.0. Where do the switches happen?
(Look at the turbid patch and at places where fish pass close to each other.)"""),
C('''tracks = run_tracker(dets)
fig, ax = plt.subplots(figsize=(9, 5)); ax.axvspan(260, 380, color="#7a8b6f", alpha=0.25)
cols = plt.cm.tab20(np.linspace(0, 1, max(len(tracks), 1)))
for c, tr in zip(cols, tracks):
    h = np.array(tr["hist"]); ax.plot(h[:, 1], h[:, 2], "-", color=c, lw=1.5); ax.text(h[0, 1], h[0, 2], str(tr["id"]), fontsize=7)
ax.set_xlim(0, W_PX); ax.set_ylim(H_PX, 0); ax.set_aspect("equal"); ax.set_title(f"{len(tracks)} confirmed tracks (colour = track ID)"); plt.show()'''),
M("""## 3. Tuning `max_age` and the gate ▶

`max_age` sets how long a track may coast. Too short and a fish that crosses the turbid patch comes out with a new
ID; too long and a coasting track can grab the wrong fish."""),
C('''rows = []
for max_age in (2, 5, 10, 20, 40):
    for gate in (15, 25, 40):
        s = score(run_tracker(dets, max_age=max_age, gate=gate), 8); s.update(max_age=max_age, gate=gate); rows.append(s)
tune = pd.DataFrame(rows)
tune.pivot(index="max_age", columns="gate", values="id_switches")'''),
M("""## 4. Camera siting: how far may a fish move between frames? ▶

Tracking gets ambiguous when a fish moves, between two frames, a large fraction of the distance to its nearest
neighbour. That depends on swimming speed, frame rate, and how many pixels the field of view spans.

**Project link.** The Aquatic Guardian team found a tracker producing 205 tracks for 12 fish because simulated fish
moved about 34 pixels per frame in a 37 cm field of view. It was documented as a **rig requirement** (frame rate and
field of view), not hidden by tuning. Reproduce the effect: same fish, sampled every `k`-th frame."""),
C('''P_fine, _ = simulate_school(n_fish=12, T=200 * 30, speed=2.0, seed=5)
nn_dist = np.median([np.sort(np.linalg.norm(P_fine[t][:, None] - P_fine[t][None], axis=2), axis=1)[:, 1] for t in range(0, 6000, 50)])
rows = []
for k in (1, 3, 6, 10, 15, 20, 30):
    Pk = P_fine[::k][:200]; d = observe(Pk, turbid_x=None, seed=2)      # always 200 frames, so runs are comparable
    s = score(run_tracker(d, gate=max(25, 2.5 * 2.0 * k), max_age=10), 12)
    s.update(px_per_frame=2.0 * k, ratio_to_spacing=round(2.0 * k / nn_dist, 2)); rows.append(s)
siting = pd.DataFrame(rows)[["px_per_frame", "ratio_to_spacing", "confirmed_tracks", "id_switches", "mean_purity"]]
print(f"median distance to nearest neighbour: {nn_dist:.0f} px"); siting'''),
M("""✏️ **Interpret:** at what ratio of *movement per frame* to *neighbour spacing* does tracking break down here?
Turn that into a requirement for a real camera: given a fish speed (cm/s) and a field of view (cm across),
what minimum frame rate keeps you below that ratio? Treat the number as a rule of thumb from this simulation,
to be checked on real footage.

## 5. Behaviour from tracks: school polarisation ▶

**Polarisation** measures how aligned a group is: average the fish's unit heading vectors and take the length.
1 means all swimming the same way, near 0 means random headings.

Velocities can come from **raw frame differencing** of a track's detections, or from the **Kalman state**. Before
running: which do you expect to be closer to the truth, and why? (Differencing amplifies position noise; the Kalman
filter smooths it but lags behind turns.)"""),
C('''def polarisation_from(vecs):
    v = np.asarray(vecs, float); n = np.linalg.norm(v, axis=1); v = v[n > 1e-9] / n[n > 1e-9, None]
    return float(np.linalg.norm(v.mean(0))) if len(v) else np.nan

rows = []
for name, turn, align in [("calm school", 0.05, 0.3), ("moderate", 0.15, 0.15), ("erratic", 0.45, 0.0)]:
    Ps, Vs = simulate_school(n_fish=10, T=200, speed=3.0, turn=turn, align=align, seed=7)
    d = observe(Ps, sigma=1.5, seed=8); trs = run_tracker(d)
    true_pol = np.mean([polarisation_from(Vs[t]) for t in range(20, 200)])
    raw_t, kal_t = {}, {}
    for tr in trs:
        h = tr["hist"]
        for (t0, x0, y0, _), (t1, x1, y1, _), kv in zip(h, h[1:], tr["vel"][1:]):
            if t1 - t0 == 1: raw_t.setdefault(t1, []).append((x1 - x0, y1 - y0))
            if kv is not None: kal_t.setdefault(t1, []).append(kv)
    raw_pol = np.nanmean([polarisation_from(v) for t, v in raw_t.items() if t >= 20 and len(v) >= 5])
    kal_pol = np.nanmean([polarisation_from(v) for t, v in kal_t.items() if t >= 20 and len(v) >= 5])
    rows.append(dict(school=name, truth=true_pol, raw_differencing=raw_pol, kalman_velocity=kal_pol))
pol = pd.DataFrame(rows).set_index("school").round(3); pol'''),
M("""Report what the table shows, including if the "more sophisticated" option isn't better. The Aquatic Guardian team
found that Kalman-filtered velocities did **not** recover higher polarisation than raw differencing, and reported the
trade-off rather than asserting the expected result. Do your numbers agree? Change `sigma` in `observe` (detection
noise) and see whether the ranking flips.

## 6. Counting fish that cross a line ▶

Many monitoring systems count animals passing a virtual line (a fishway, a net opening). With tracks, count a
crossing when a confirmed track's position goes from one side to the other."""),
C('''LINE_X = W_PX / 2
def crossings_from_tracks(trs):
    n = 0
    for tr in trs:
        xs = np.array([h[1] for h in tr["hist"]]); n += int(np.sum(np.diff(np.sign(xs - LINE_X)) != 0))
    return n
true_cross = int(sum(np.sum(np.diff(np.sign(P[:, i, 0] - LINE_X)) != 0) for i in range(P.shape[1])))
print(f"true crossings: {true_cross}   counted from tracks: {crossings_from_tracks(run_tracker(dets))}")
print(f"counted from tracks, clear water (no turbid patch): {crossings_from_tracks(run_tracker(observe(P, seed=1)))}")'''),
M("""### 🧩 Exercise — write the polarisation metric yourself

Write `my_polarisation(velocities)` taking an `(N, 2)` array of velocity vectors and returning a number between 0
and 1. Remember to normalise each vector to unit length first (otherwise fast fish count more)."""),
C('''def my_polarisation(velocities):
    return None   # YOUR CODE

def check_pol(f):
    tests = [np.array([[1, 0], [2, 0], [5, 0]]), np.array([[1, 0], [-1, 0]]), np.array([[1, 0], [0, 3]])]
    expect = [1.0, 0.0, np.sqrt(2) / 2]
    try: got = [f(t) for t in tests]
    except Exception as e: print("Error:", e); return
    if any(g is None for g in got): print("Not yet: return a number."); return
    print("✅ Correct." if np.allclose(got, expect, atol=1e-6) else f"❌ Expected {np.round(expect, 3)}, got {np.round(got, 3)}. Did you normalise each vector?")
check_pol(my_polarisation)'''),
C('''#@title Solution (double-click to show)
def my_polarisation(velocities):
    v = np.asarray(velocities, float); v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return float(np.linalg.norm(v.mean(axis=0)))
check_pol(my_polarisation)''', hidden=True),
M("""## Check yourself

<details><summary>1. What does the Kalman filter do in a frame with no detection?</summary>It predicts (coasts) without updating; its uncertainty grows until a detection arrives.</details>
<details><summary>2. Your tracker reports 40 tracks for 10 fish. Name two likely causes.</summary>Fish move too far between frames for the gate (frame rate too low or field of view too small), or max_age too short so tracks die during occlusions or turbid patches.</details>
<details><summary>3. Why gate the assignment?</summary>Without a gate the Hungarian algorithm will always pair something, even a track with a distant false positive.</details>
<details><summary>4. A smoother velocity estimate gives a less accurate polarisation. How can that be?</summary>Smoothing lags behind turns; when headings change quickly, lagged velocities point the wrong way. Noise and lag trade off.</details>

**Next:** Module 09, *From pixels to defensible claims* — why fewer detections near an outfall doesn't mean fewer fish."""),
]
build(cells, "out/CV08_tracking_over_time.ipynb")
