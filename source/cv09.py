from nbcommon import *
OCC_SETUP = r'''#@title ▶ Setup
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 10})
print("ready")'''
cells = [
M("# Module 09 — From pixels to defensible claims\n\n" + HEADER.format(num="09", time="about 2 hours",
  skip="never. Every path through the course ends here, including the policy path.") + """
### By the end you can
1. Explain why **fewer detections** near an outfall does not show **fewer animals**.
2. Separate *occupancy* (is the animal there?) from *detection probability* (would the camera see it?) using
   repeat surveys, and fit a simple occupancy model.
3. Connect the detection side to what you measured in Modules 02, 05 and 07.
4. Write a **conditional finding** that says what the evidence supports, under what assumptions, and what it doesn't show.
5. Name the welfare, surveillance and licensing questions that come with pointing cameras at sentient animals.

This is the reasoning behind the monitoring notebook's correction of occupancy for detection probability, and the
rule in every project notebook that a detection result is never, on its own, an ecological-safety claim.
"""),
C(OCC_SETUP, hidden=True),
M("""## 1. A world with no impact at all ▶

150 survey sites along a line running away from a warm-water outfall. **Every site has the same true occupancy:
a 60% chance that a given fish species uses it.** The outfall does nothing to the fish.

Turbidity has two parts: a plume component that is higher near the outfall, and a weather and tide component that
changes from visit to visit everywhere. The camera's chance of detecting the fish on a visit, *when it is there*,
falls as turbidity rises (Module 02: shorter detection range). Each site is visited 6 times."""),
C('''def simulate_sites(n_sites=150, visits=6, psi_far=0.6, psi_near=0.6, plume=3.0, seed=0):
    r = np.random.default_rng(seed)
    dist_m = np.sort(r.uniform(0, 3000, n_sites))
    near = np.exp(-dist_m / 700)                                      # 1 at the outfall, about 0 far away
    psi = psi_far + (psi_near - psi_far) * near                       # true occupancy at each site
    occupied = r.random(n_sites) < psi
    weather = r.gamma(2.0, 0.6, (n_sites, visits)) * 1.5              # varies visit to visit, everywhere
    turb = 1.0 + plume * near[:, None] + weather                      # illustrative turbidity units
    p = expit(2.0 - 0.7 * turb)                                       # detection probability per visit
    y = ((r.random((n_sites, visits)) < p) & occupied[:, None]).astype(int)
    sites = pd.DataFrame(dict(dist_m=dist_m, near=near, psi_true=psi, occupied=occupied))
    return sites, turb, p, y

sites, turb, p, y = simulate_sites(seed=0)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].plot(sites.dist_m, turb.mean(1), "o", ms=3); ax[0].set_xlabel("distance from outfall (m)"); ax[0].set_ylabel("mean turbidity (illustrative)")
ax[1].plot(sites.dist_m, p.mean(1), "o", ms=3, color="#b5651d"); ax[1].set_xlabel("distance from outfall (m)"); ax[1].set_ylabel("detection probability per visit")
plt.tight_layout(); plt.show()'''),
M("""## 2. The naive analysis ▶

Two common summaries: the share of visits with a detection, and the share of sites where the fish was ever detected."""),
C('''bins = pd.cut(sites.dist_m, [0, 1000, 2000, 3000], labels=["0-1 km", "1-2 km", "2-3 km"])
naive = pd.DataFrame({"true occupancy": sites.psi_true, "visits with a detection": y.mean(1),
                      "sites ever detected": y.max(1)}).groupby(bins, observed=True).mean().round(2)
naive'''),
M("""Read the detection columns on their own and it looks like an impact near the outfall. It's an artefact: the camera
saw less, the fish didn't leave. **This is the most consequential mistake in the course**, because it's the one most
likely to end up in a permit decision.

## 3. Separating "there" from "seen": an occupancy model ▶

Repeat visits are the key. At an occupied site, a fish detected on some visits and missed on others tells us about
detection probability. The site-level likelihood (MacKenzie et al., 2002) is

$$L_i = \\psi_i \\prod_j p_{ij}^{\\,y_{ij}} (1 - p_{ij})^{1 - y_{ij}} \\;+\\; (1 - \\psi_i)\\,\\mathbb{1}[\\text{no detections at } i]$$

The first term: the site is occupied and we saw this exact pattern. The second: unoccupied, which is only possible
if we never detected anything. We let occupancy depend on closeness to the outfall and detection depend on each
visit's turbidity, fit both by maximum likelihood, and get uncertainty from the curvature of the likelihood.

**Why this works here:** turbidity changes from visit to visit at every site (weather, tide). That within-site
variation lets the model learn how turbidity affects detection separately from where the site is. If turbidity were
fixed by distance alone, the two effects would be inseparable, and no model could untangle them. Survey design
decides what's knowable."""),
C('''def neg_log_lik(theta, near, turb, y):
    b0, b1, a0, a1 = theta
    psi = expit(b0 + b1 * near); pj = expit(a0 + a1 * turb)
    lik_occ = np.prod(np.where(y == 1, pj, 1 - pj), axis=1)
    return -np.sum(np.log(psi * lik_occ + (1 - psi) * (y.sum(1) == 0) + 1e-300))

def hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n)); E = np.eye(n) * h
    for i in range(n):
        for j in range(n):
            H[i, j] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j]) - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return H

def fit_occupancy(sites, turb, y):
    f = lambda th: neg_log_lik(th, sites.near.values, turb, y)
    res = minimize(f, np.zeros(4), method="BFGS"); cov = np.linalg.inv(hessian(f, res.x))
    se = np.sqrt(np.diag(cov))
    def psi_at(near_value):
        x = np.array([1.0, near_value]); eta = x @ res.x[:2]; s = np.sqrt(x @ cov[:2, :2] @ x)
        return expit(eta), expit(eta - 1.96 * s), expit(eta + 1.96 * s)
    return dict(theta=res.x, se=se, psi_at=psi_at)

fit = fit_occupancy(sites, turb, y)
b1, se1 = fit["theta"][1], fit["se"][1]
print(f"outfall effect on occupancy (logit scale): {b1:+.2f}, 95% interval {b1 - 1.96 * se1:+.2f} to {b1 + 1.96 * se1:+.2f}   (truth 0)")
print(f"turbidity effect on detection (logit):     {fit['theta'][3]:+.2f} ± {1.96 * fit['se'][3]:.2f}            (truth -0.70)")
for label, nv in [("at the outfall", 1.0), ("far away", 0.0)]:
    est, lo, hi = fit["psi_at"](nv); print(f"occupancy {label:15s}: {est:.2f} (95% interval {lo:.2f}-{hi:.2f})   truth 0.60")'''),
M("""✏️ **Interpret:** does the interval for the outfall effect include 0? Then the data are consistent with no impact
once detection is accounted for, which is the truth here. Note how wide the occupancy intervals are, even with 150
sites and 900 visits: honest uncertainty is part of the finding.

## 4. Repeat the survey many times: false alarms and power ▶

One survey can be lucky. Simulate 20 surveys in each of two worlds: **no impact**, and a **real impact** where
occupancy at the outfall falls from 0.6 to 0.25. Count how often each analysis concludes "impact":

- **naive**: visit detection rate within 1 km is at least 25% lower than beyond 2 km;
- **model**: the 95% interval for the outfall effect lies entirely below 0."""),
C('''rows = []
for world, psi_near in [("no impact", 0.6), ("real impact", 0.25)]:
    for seed in range(20):
        s_, t_, p_, y_ = simulate_sites(psi_near=psi_near, seed=100 + seed)
        near_rate, far_rate = y_[s_.dist_m < 1000].mean(), y_[s_.dist_m > 2000].mean()
        f_ = fit_occupancy(s_, t_, y_); upper = f_["theta"][1] + 1.96 * f_["se"][1]
        rows.append(dict(world=world, naive_says_impact=near_rate < 0.75 * far_rate, model_says_impact=upper < 0))
power = pd.DataFrame(rows).groupby("world")[["naive_says_impact", "model_says_impact"]].mean()
power.rename(columns={"naive_says_impact": "naive: share of surveys", "model_says_impact": "model: share of surveys"})'''),
M("""Read the **no impact** row first: how often does each method raise a false alarm? Then the **real impact** row:
how often does each catch the real drop in occupancy? (That's the survey's **power**.)

Two lessons. The naive analysis "finds" an impact whether or not there is one, so it carries no information here. The
model rarely raises a false alarm, but a survey of this size misses a real effect a good share of the time. More
sites or visits, or a camera with better detection in turbid water, raises power. Report power with any finding of
"no detectable effect"; absence of evidence is only meaningful if the survey could have seen the effect."""),
M("""## 5. Where the detection side comes from

Everything you built earlier feeds the detection model:

| Course module | What it tells the occupancy model |
|---|---|
| 02, image formation | detection range falls with turbidity and distance |
| 05, robustness by condition | recall by condition **is** a detection probability |
| 07, detection AP and max recall | the ceiling on what a detector can ever see in given water |
| 08, tracking | double counts and missed tracks bias counts in both directions |

The thermal-tidal notebook supplies the other half: modelled temperature, dissolved oxygen and plume extent become
**site and visit covariates** here, rather than assumptions.

## 6. Writing a conditional finding

A finding should say what the evidence supports, under which assumptions, and what it does **not** show. Template:

> *Under [assumptions: survey design, detection model, covariates], [estimate] was [value] ([interval]).
> This [does / does not] indicate [effect]. It does not show [what's outside the evidence].
> It depends on [the inputs that are placeholders or unvalidated].*

**Weak:** "Camera surveys found far fewer fish near the outfall, indicating an ecological impact."

The cell below writes the defensible version from **your** fitted numbers, so the wording can't drift from the evidence."""),
C('''near_rate, far_rate = y[sites.dist_m < 1000].mean(), y[sites.dist_m > 2000].mean()
lo1, hi1 = b1 - 1.96 * se1, b1 + 1.96 * se1
verdict = ("consistent with no change in occupancy" if lo1 <= 0 <= hi1 else
           "indicating lower occupancy near the outfall" if hi1 < 0 else "indicating higher occupancy near the outfall")
print(f"Visits within 1 km of the outfall recorded a detection {100 * (1 - near_rate / far_rate):.0f}% less often than "
      f"visits beyond 2 km. After modelling detection probability as a function of each visit's turbidity "
      f"({y.shape[1]} visits at each of {y.shape[0]} sites), the estimated outfall effect on occupancy was {b1:+.2f} on the "
      f"logit scale (95% interval {lo1:+.2f} to {hi1:+.2f}), {verdict}. This does not show that the outfall has no effect "
      f"on fish health, behaviour, or species not surveyed, and a survey of this size can miss a real effect (section 4). "
      f"Detection parameters come from synthetic data and must be re-estimated on site footage.")'''),
M("""## 7. Responsibilities that come with the camera

- **Sentience.** The UK's Animal Welfare (Sentience) Act 2022, an enacted law, recognises vertebrates (fish included),
  decapod crustaceans and cephalopod molluscs as sentient for the purposes of that Act, following a 2021 evidence
  review led by Jonathan Birch at LSE. A welfare flag from a model is a reason for a person to look, never a diagnosis.
- **The setup can change the result.** Lights, noise and equipment can change animal behaviour. Record the rig.
- **Surveillance and dual use.** The same tracking that counts fish can follow people on a deck or a shoreline.
  Ask who is recorded, whether they know, and how long footage is kept.
- **Licences.** Record each dataset's licence and each tool's licence (Ultralytics YOLO is AGPL-3.0) before use.
- **Policy accuracy.** When findings are cited in policy work, keep enacted law, proposed bills and guidance clearly apart.
- **Human review.** Low confidence, degraded images, and anything with regulatory consequences go to a person, by default."""),
M("""### 🧩 Exercise — how many visits?

If the per-visit detection probability at an occupied site is $p = 0.25$, the chance of at least one detection in $K$
visits is $1 - (1 - p)^K$. Find the smallest $K$ that gives at least a 90% chance. Store it in `visits_needed`."""),
C('''visits_needed = None   # YOUR CODE (an integer)

def check_visits(k):
    exp = int(np.ceil(np.log(0.1) / np.log(0.75)))
    if k is None: print("Not yet: replace None."); return
    print(f"✅ Correct: {exp} visits (chance {1 - 0.75 ** exp:.3f})." if int(k) == exp else f"❌ Expected {exp}. With K={k}: {1 - 0.75 ** int(k):.3f}")
check_visits(visits_needed)'''),
C('''#@title Solution (double-click to show)
visits_needed = next(k for k in range(1, 100) if 1 - 0.75 ** k >= 0.9)
check_visits(visits_needed)''', hidden=True),
M("""### Write your own finding (not graded)

Using any result from this track (detection range, robustness by condition, AP, tracking switches or occupancy),
write one conditional finding with the template in section 6. Paste it into your capstone report."""),
M("""## Check yourself

<details><summary>1. Detections fall by half near an outfall. List two explanations other than fewer fish.</summary>Lower detection probability (turbidity, light, plume), changed behaviour (fish avoid the camera's field of view but not the site), different survey effort, or a model performing worse in those conditions.</details>
<details><summary>2. Why do repeat visits make detection probability estimable?</summary>At sites where the animal is present, the pattern of hits and misses across visits shows how often it's missed.</details>
<details><summary>3. Your occupancy model's interval for the outfall effect is -1.9 to +1.3. What can you say?</summary>The data are consistent with no effect, a negative effect or a positive one. The survey can't distinguish them; you need more sites or visits, or better detection.</details>
<details><summary>4. A model flags a fish as "abnormal". What happens next?</summary>A person reviews it. Model output is a reason to look, not a diagnosis or a regulatory finding.</details>

**You've finished the core track.** Go to the capstones on the course page to apply this to the project notebooks."""),
]
build(cells, "out/CV09_from_pixels_to_claims.ipynb")
