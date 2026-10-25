from nbcommon import *
cells = [
M("# Module 05 — Robustness, shift, and honest reporting\n\n" + HEADER.format(num="05", time="about 2 hours",
  skip="you routinely report metrics by condition and by site, and write data and model cards.") + """
### By the end you can
1. Report performance **by condition** (clear, turbid, low light, blur, far away), not as one number.
2. Measure **distribution shift**: train in one habitat, test in another.
3. Use **augmentation** and say honestly what it fixes and what it doesn't.
4. Build an **image-quality gate** that routes unusable frames to `unknown` instead of guessing.
5. Write an error taxonomy, a data card and a model card, and emit a conservative **event record**.

This module covers **Mentee 1, Tasks 1, 8, 9 and 10**, and the event schema in the **Weeks 1–4 scaffold**.
"""),
C(SETUP, hidden=True),
C(FRAMES, hidden=True),
C('''import pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def features(img, fg_thresh=0.10):            # same features as Module 04
    corners = np.concatenate([img[:6, :6].reshape(-1, 3), img[:6, -6:].reshape(-1, 3), img[-6:, :6].reshape(-1, 3), img[-6:, -6:].reshape(-1, 3)])
    bgc = np.median(corners, 0); mask = np.sqrt(((img - bgc) ** 2).sum(-1)) > fg_thresh; area = mask.mean()
    if mask.sum() >= 5:
        ys, xs = np.where(mask); ev = np.clip(np.linalg.eigvalsh(np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))), 1e-6, None)
        elong = np.sqrt(ev[1] / ev[0]); mr, mg, mb = img[mask].mean(0) - bgc
    else: elong = mr = mg = mb = 0.0
    g = img.mean(-1); edge = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()
    return np.array([area, elong, mr, mg, mb, edge])
feat = lambda imgs: np.array([features(im) for im in imgs])

Xtr, ytr, str_, htr = make_clips(n_sessions_per_class=36, seed=10)      # training sessions
Xte, yte, ste, hte = make_clips(n_sessions_per_class=15, seed=99)        # NEW sessions for testing
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(Xtr), ytr)
print(f"clean test accuracy on new sessions: {model.score(feat(Xte), yte):.3f}")'''),
M("""## 1. One number hides everything: evaluate by condition ▶

We apply the project's `degrade()` utility to the test frames, and also generate genuinely **farther** objects with
the physics model from Module 02. Same model, same test sessions, different conditions."""),
C('''HABITATS["reef_far"] = dict(HABITATS["reef"], dist=(5.0, 8.0))
HABITATS["harbour_far"] = dict(HABITATS["harbour"], dist=(2.5, 4.0))
Xfar, yfar, _, _ = make_clips(n_sessions_per_class=15, habitats=("reef_far", "harbour_far"), seed=7)

CONDITIONS = {
    "clean": dict(), "turbid (degrade 0.3)": dict(turbidity=0.3), "low light (degrade 0.5)": dict(low_light=0.5),
    "blur (k=3)": dict(blur_k=3), "turbid + blur": dict(turbidity=0.3, blur_k=3),
}
def by_condition(m):
    rows = [dict(condition=k, accuracy=m.score(feat([degrade(im, **kw) for im in Xte]), yte)) for k, kw in CONDITIONS.items()]
    rows.append(dict(condition="farther objects (physics)", accuracy=m.score(feat(Xfar), yfar)))
    return pd.DataFrame(rows).set_index("condition")["accuracy"]
rob = by_condition(model)
ax = rob.plot.barh(figsize=(6, 2.8), color="#2f6f8f"); ax.set_xlim(0, 1); ax.set_xlabel("accuracy"); ax.invert_yaxis()
ax.set_title("Same model, six conditions"); plt.tight_layout(); plt.show()
rob.round(3)'''),
M("""✏️ **Interpret:** which condition hurts most? Look back at the `features()` function and explain the mechanism in
one sentence (what does turbidity do to the foreground mask?).

## 2. Distribution shift: a new habitat ▶

Train only on clear **reef** clips, then test on **harbour** clips. This is what happens when a model built on a
public dataset from one coast is pointed at a different site."""),
C('''def split(X, y, h, habs):
    sel = np.isin(h, habs); return X[sel], y[sel]
rows = []
for train_habs in [("reef",), ("reef", "kelp"), ("reef", "kelp", "harbour")]:
    Xa, ya = split(Xtr, ytr, htr, train_habs)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(Xa), ya)
    for test_hab in ("reef", "kelp", "harbour"):
        Xb, yb = split(Xte, yte, hte, (test_hab,))
        rows.append(dict(trained_on="+".join(train_habs), tested_on=test_hab, accuracy=m.score(feat(Xb), yb)))
shift = pd.DataFrame(rows).pivot(index="trained_on", columns="tested_on", values="accuracy").round(3)
shift'''),
M("""Read the table across each row. The gap between "tested on what you trained on" and "tested somewhere new" is
the number you must report, because the project's cameras will not sit on the reef your dataset came from.

## 3. Augmentation ▶

Add degraded copies of training frames so the model sees turbid and dark examples during training."""),
C('''r = np.random.default_rng(3); aug_X, aug_y = list(Xtr), list(ytr)
for im, c in zip(Xtr, ytr):
    kw = [dict(turbidity=0.35), dict(low_light=0.5), dict(turbidity=0.25, blur_k=3), dict(blur_k=3)][r.integers(4)]
    aug_X.append(degrade(im, **kw)); aug_y.append(c)
model_aug = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(feat(aug_X), np.array(aug_y))
comparison = pd.DataFrame({"baseline": rob, "augmented": by_condition(model_aug)}).round(3)
comparison'''),
M("""Augmentation usually recovers some degraded-condition accuracy, and often costs some clean accuracy: a 6-feature
linear model has little room to learn both. Report both columns, not just the one that improved.

Now look at the last row. Augmenting with `degrade()` barely helps with objects that are physically farther away.
Training on one degradation model doesn't buy robustness to a different one, which is why Module 02 asked you to
know what `degrade()` assumes.

## 4. An image-quality gate ▶

The scaffold's event schema labels **image quality** (`good | degraded | unusable`) separately from the organism,
so a bad frame is never silently treated as a confident detection. We score quality from three cheap statistics
and set limits from the **training** frames."""),
C('''from scipy import ndimage
RING = np.zeros((HW, HW), bool); RING[:6] = RING[-6:] = True; RING[:, :6] = RING[:, -6:] = True
def quality_stats(img):
    """Measured on the 6-pixel border ring, where there is usually only background. That keeps the gate about
    the IMAGE, not about what's in it (otherwise empty frames would look 'low contrast' and get flagged)."""
    g = img.mean(-1); lap = ndimage.laplace(ndimage.gaussian_filter(g, 0.5))
    return dict(brightness=g[RING].mean(), texture=g[RING].std(), sharpness=lap[RING].var())
qs_train = pd.DataFrame([quality_stats(im) for im in Xtr])
LIMITS = qs_train.quantile(0.02)          # anything below the 2nd percentile of training frames is out of range
print("limits (2nd percentile of training frames):"); print(LIMITS.round(5).to_string())

def quality(img):
    q = quality_stats(img); low = [k for k in LIMITS.index if q[k] < LIMITS[k]]
    return "unusable" if len(low) >= 2 else ("degraded" if len(low) == 1 else "good")

rows = []
for k, kw in CONDITIONS.items():
    imgs = [degrade(im, **kw) for im in Xte]; q = np.array([quality(im) for im in imgs])
    pred = model.predict(feat(imgs)); ok = q == "good"
    rows.append(dict(condition=k, share_good=ok.mean(), acc_on_good=(pred[ok] == yte[ok]).mean() if ok.any() else np.nan,
                     acc_on_flagged=(pred[~ok] == yte[~ok]).mean() if (~ok).any() else np.nan))
pd.DataFrame(rows).set_index("condition").round(3)'''),
M("""✏️ **Interpret:** three things to look for.

- Where some frames pass and some are flagged, is accuracy lower on the flagged ones? Then the gate is routing the
  riskiest frames to `unknown`/human review, which is its job.
- Where the gate flags *every* frame (`NaN` under `acc_on_good`), the whole condition goes to review. Safe, but costly.
- In clean water the gate still flags a few frames that the model would have got right. That's its false-alarm cost.
  Write all three into your report.

## 5. Error taxonomy ▶

Bucket every mistake by (true class, predicted class, condition). Patterns here tell you what data to collect next."""),
C('''errs = []
for k, kw in CONDITIONS.items():
    imgs = [degrade(im, **kw) for im in Xte]; pred = model.predict(feat(imgs))
    for t, p in zip(yte, pred):
        if t != p: errs.append(dict(true=CLASSES[t], predicted=CLASSES[p], condition=k))
taxonomy = pd.DataFrame(errs).value_counts().rename("count").reset_index().head(10)
taxonomy'''),
M("""## 6. Data card, model card, and a conservative event record ▶

These go in the repo with the code. The event record mirrors the Weeks 1–4 scaffold: when evidence is weak or the
frame is degraded, the defaults are `unknown` and `human_review_required`, **never** a "safe" state."""),
C('''data_card = {
    "name": "Synthetic staged clips (teaching)", "source": "generated in this notebook", "licence": "n/a (synthetic)",
    "classes": CLASSES, "split": "by session (clip), never by frame",
    "habitats": sorted(set(htr)), "known_gaps": ["no real organisms", "no occlusion between animals", "fixed camera"],
}
model_card = {
    "model": "standardised 6-feature logistic regression", "trained_on": data_card["name"],
    "metrics_by_condition": rob.round(3).to_dict(), "metrics_by_habitat_shift": shift.to_dict(),
    "intended_use": "teaching; screening frames for human review",
    "must_not_be_used_for": None,            # 🧩 fill this in (a list of strings)
}
def event_record(img, model):
    q = quality(img)
    proba = model.predict_proba(feat([img]))[0]; top = int(proba.argmax())
    confident = (q == "good") and proba[top] >= 0.8
    return {
        "image_quality": q, "detections": [dict(cls=CLASSES[top], confidence=round(float(proba[top]), 3))],
        "event_type": "normal_activity" if confident else "unknown",
        "action": "log" if confident else "human_review_required",
    }
print(json.dumps(event_record(Xte[0], model), indent=1))
print(json.dumps(event_record(degrade(Xte[0], turbidity=0.7, blur_k=7), model), indent=1))'''),
M("""### 🧩 Exercise — the most important field

Fill in `model_card["must_not_be_used_for"]` with at least two specific uses this model must not support. One should
concern ecological or regulatory claims."""),
C('''def check_card(card):
    v = card.get("must_not_be_used_for")
    if not v: print("Not yet: add a list of at least two strings."); return
    text = " ".join(v).lower()
    if len(v) >= 2 and any(w in text for w in ("ecolog", "safe", "permit", "regulat", "impact", "health", "compliance")):
        print("✅ Good. Keep this list in the repo next to the metrics.")
    else: print("Add at least two items, including one about ecological, safety or regulatory claims.")
check_card(model_card)'''),
C('''#@title Example answer (double-click to show)
model_card["must_not_be_used_for"] = [
    "claiming an ecosystem is healthy or unharmed (absence of detections is not absence of animals)",
    "permit or compliance decisions without detection-probability correction and human review",
    "species identification (classes are fish / non_fish / empty only)",
    "sites or water conditions outside the evaluated habitats without re-validation",
]
check_card(model_card)
with open("model_card.json", "w") as f: json.dump(model_card, f, indent=1, default=str)
print("saved model_card.json")''', hidden=True),
M("""## Check yourself

<details><summary>1. Your model has 0.95 accuracy overall and 0.55 in turbid water. What's the headline?</summary>Both numbers, with the condition named. The overall figure alone would mislead anyone deploying the model where turbidity is common.</details>
<details><summary>2. Why set quality limits from training frames rather than test frames?</summary>The gate must be fixed before seeing test data, otherwise the evaluation is tuned to the test set.</details>
<details><summary>3. What should an event record say when the frame is degraded and confidence is 0.62?</summary>event_type "unknown", action "human_review_required". Never a safe or normal state by default.</details>
<details><summary>4. Augmentation raised turbid accuracy from 0.5 to 0.7. Is the turbidity problem solved?</summary>No. It's reduced. Report the remaining gap and the clean-accuracy change, and keep the quality gate.</details>

**Next:** Module 06, *Neural networks and CNNs* — a model that learns its own features, and whether that helps under turbidity."""),
]
build(cells, "out/CV05_robustness_and_reporting.ipynb")
