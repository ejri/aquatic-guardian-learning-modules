from nbcommon import *
cells = [
M("# Module 04 — Learning to classify, and evaluating without fooling yourself\n\n" + HEADER.format(num="04", time="about 2.5 hours",
  skip="you know train/validation/test splits, data leakage, precision/recall/F1, and confidence thresholds.") + """
### By the end you can
1. Turn images into features and train a softmax (multinomial logistic) classifier from scratch.
2. Split video data **by clip/session**, and show what happens to your score if you don't.
3. Read a confusion matrix and compute precision, recall and F1 per class.
4. Explain why accuracy misleads on imbalanced data.
5. Add an **abstention** rule that returns `unknown` below a confidence threshold, and read the coverage trade-off.

This module is the backbone of **Mentee 1, Tasks 2–5, 7 and 9**.
"""),
C(SETUP, hidden=True),
C(FRAMES, hidden=True),
C('''import pandas as pd
X, y, sess, hab = make_clips(n_sessions_per_class=45, frames_per_clip=12, seed=0)
print("frames:", X.shape, "sessions:", len(np.unique(sess)))
print(pd.crosstab(pd.Series(hab, name="habitat"), pd.Series([CLASSES[i] for i in y], name="class")))
fig, ax = plt.subplots(1, 8, figsize=(12, 1.9))
first = np.where(sess == 3)[0][:8]
for a, k in zip(ax, first): a.imshow(X[k]); a.axis("off")
fig.suptitle("Eight consecutive frames from ONE clip: they are nearly identical", fontsize=9); plt.show()'''),
M("""## 1. The leakage trap ▶

Consecutive frames from one clip are near-duplicates. If some frames of a clip land in training and others in
testing, the test is partly a memory test. We show this with the laziest possible classifier,
**1-nearest-neighbour on raw pixels**: it labels a test frame with the label of the most similar training frame."""),
C('''from sklearn.neighbors import KNeighborsClassifier
flat = X[:, ::2, ::2].reshape(len(X), -1)              # downsample to 24x24 for speed
r = np.random.default_rng(1)

# (a) random split by FRAME
idx = r.permutation(len(X)); cut = int(0.7 * len(X))
tr_f, te_f = idx[:cut], idx[cut:]
# (b) split by SESSION (clip)
sessions = np.unique(sess); r.shuffle(sessions)
train_sessions = set(sessions[:int(0.7 * len(sessions))])
tr_s = np.array([i for i in range(len(X)) if sess[i] in train_sessions]); te_s = np.array([i for i in range(len(X)) if sess[i] not in train_sessions])

knn = KNeighborsClassifier(1)
acc_frame = knn.fit(flat[tr_f], y[tr_f]).score(flat[te_f], y[te_f])
acc_sess = knn.fit(flat[tr_s], y[tr_s]).score(flat[te_s], y[te_s])
print(f"1-NN accuracy, random split by frame:   {acc_frame:.3f}")
print(f"1-NN accuracy, split by session (clip): {acc_sess:.3f}")'''),
M("""The frame-split number is the one that gets reported by mistake. The session split is the honest estimate of
how the model does on a **new** camera visit. Rule for the whole project: **split by clip, session, cage or site —
never by frame.** From here on we use the session split.

## 2. Features: describing a frame with a few numbers ▶

The Mentee 1 baseline computes interpretable features: foreground **area**, **elongation**, mean **colour** of the
foreground, and **edge density**. Same idea here (our frames are dimmer, so the foreground threshold is lower)."""),
C('''def features(img, fg_thresh=0.10):
    corners = np.concatenate([img[:6, :6].reshape(-1, 3), img[:6, -6:].reshape(-1, 3),
                              img[-6:, :6].reshape(-1, 3), img[-6:, -6:].reshape(-1, 3)])
    bgc = np.median(corners, 0)
    mask = np.sqrt(((img - bgc) ** 2).sum(-1)) > fg_thresh
    area = mask.mean()
    if mask.sum() >= 5:
        ys, xs = np.where(mask); ev = np.clip(np.linalg.eigvalsh(np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))), 1e-6, None)
        elong = np.sqrt(ev[1] / ev[0]); mr, mg, mb = img[mask].mean(0) - bgc
    else:
        elong = mr = mg = mb = 0.0
    g = img.mean(-1); edge = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()
    return np.array([area, elong, mr, mg, mb, edge])
FEATS = ["area", "elongation", "dR", "dG", "dB", "edge"]

F = np.array([features(im) for im in X])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for ci, c in enumerate(CLASSES):
    ax[0].scatter(F[y == ci, 0], F[y == ci, 1], s=6, alpha=0.5, label=c)
ax[0].set_xlabel("area (fraction of frame)"); ax[0].set_ylabel("elongation"); ax[0].legend(fontsize=8)
ax[0].set_title("Two features already separate most frames")
pd.DataFrame(F, columns=FEATS).assign(cls=[CLASSES[i] for i in y]).groupby("cls").mean().round(3).plot.bar(ax=ax[1], legend=True)
ax[1].set_title("mean feature value by class"); ax[1].tick_params(axis="x", rotation=0); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()'''),
M(r"""## 3. A softmax classifier from scratch ▶

For each class $k$ the model computes a score $z_k = \mathbf{w}_k \cdot \mathbf{x} + b_k$, then turns scores into
probabilities with softmax, $p_k = e^{z_k} / \sum_j e^{z_j}$. Training nudges $\mathbf{w}$ and $b$ downhill on the
**cross-entropy** loss, $-\log p_{\text{true class}}$, averaged over frames. Features are standardised first
(mean 0, standard deviation 1) using **training statistics only**."""),
C('''Ftr, Fte, ytr, yte = F[tr_s], F[te_s], y[tr_s], y[te_s]
mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
Ztr, Zte = (Ftr - mu) / sd, (Fte - mu) / sd

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def train_softmax(Xs, ys, C=3, lr=0.5, epochs=400, l2=1e-3):
    n, d = Xs.shape; W = np.zeros((d, C)); b = np.zeros(C); Y = np.eye(C)[ys]; losses = []
    for _ in range(epochs):
        P = softmax(Xs @ W + b)
        losses.append(-np.log(P[np.arange(n), ys] + 1e-12).mean())
        W -= lr * (Xs.T @ (P - Y) / n + l2 * W); b -= lr * (P - Y).mean(0)
    return W, b, losses

W, b, losses = train_softmax(Ztr, ytr)
P = softmax(Zte @ W + b); pred = P.argmax(1)
plt.figure(figsize=(5, 2.6)); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("Learning curve"); plt.show()
print(f"test accuracy (session split): {(pred == yte).mean():.3f}")

from sklearn.linear_model import LogisticRegression
sk = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
print(f"scikit-learn LogisticRegression on the same features: {sk.score(Zte, yte):.3f}")'''),
M("""## 4. The confusion matrix and per-class metrics ▶

Rows are the truth, columns the prediction. For one class:

- **Precision** = of the frames we *called* fish, how many were fish? (cost of false alarms)
- **Recall** = of the frames that *were* fish, how many did we find? (cost of misses)
- **F1** = their harmonic mean."""),
C('''def confusion(y_true, y_pred, n=3):
    M = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred): M[t, p] += 1
    return M

def class_metrics(M):
    rows = {}
    for i, c in enumerate(CLASSES):
        tp = M[i, i]; fp = M[:, i].sum() - tp; fn = M[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        rows[c] = dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, support=int(M[i].sum()))
    return pd.DataFrame(rows).T.round(3)

Mx = confusion(yte, pred)
fig, ax = plt.subplots(figsize=(4, 3.4)); ax.imshow(Mx, cmap="Blues")
for i in range(3):
    for j in range(3): ax.text(j, i, Mx[i, j], ha="center", va="center", color="white" if Mx[i, j] > Mx.max() / 2 else "black")
ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, rotation=25); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); plt.tight_layout(); plt.show()
class_metrics(Mx)'''),
M("""✏️ **Interpret:** which error is more costly for each use?

- a **ship-strike alert** for a protected species (miss = animal hit; false alarm = vessel slows);
- an **aquaculture welfare flag** that sends a person to check a fish;
- a **population count** used in a permit decision.

There's no single right threshold. Write down which metric you'd prioritise for each, and why.

## 5. Why accuracy lies on imbalanced data ▶

Real camera traps are mostly empty. Below, a test set where 85% of frames are empty, and a "model" that always
says `empty_frame`."""),
C('''r = np.random.default_rng(5)
empty_idx = te_s[yte == 2]; other_idx = te_s[yte != 2]
keep_other = r.choice(other_idx, size=int(len(empty_idx) * 0.15 / 0.85), replace=False)
imb = np.concatenate([empty_idx, keep_other]); y_imb = y[imb]
lazy = np.full_like(y_imb, 2)
ours = softmax(((F[imb] - mu) / sd) @ W + b).argmax(1)
for name, p in [("always 'empty_frame'", lazy), ("our softmax model", ours)]:
    m = class_metrics(confusion(y_imb, p))
    print(f"{name:22s} accuracy {(p == y_imb).mean():.3f}   macro-F1 {m.f1.mean():.3f}   fish recall {m.loc['fish', 'recall']:.3f}")'''),
M("""Report **per-class** recall/precision and a **macro** average (each class counts equally), not just accuracy.

## 6. Abstention: saying "I don't know" ▶

The model's top probability is a (rough) confidence. Below a threshold we return `unknown` and send the frame for
human review instead of acting on weak evidence. **Coverage** is the fraction of frames we still decide on. This is
Mentee 1's Task 7, and it's how the project's safety rules avoid treating a weak detection as fact."""),
C('''maxp = P.max(1); rows = []
for thr in [0.34, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    keep = maxp >= thr
    rows.append(dict(threshold=thr, coverage=keep.mean(), accuracy_on_kept=(pred[keep] == yte[keep]).mean() if keep.any() else np.nan,
                     sent_to_review=int((~keep).sum())))
abst = pd.DataFrame(rows).round(3)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(abst.threshold, abst.coverage, "o-", label="coverage"); ax.plot(abst.threshold, abst.accuracy_on_kept, "s-", label="accuracy on kept")
ax.set_xlabel("confidence threshold"); ax.legend(); ax.set_title("Abstention trade-off"); plt.tight_layout(); plt.show()
abst'''),
M("""**Caution.** A classifier's probabilities are not automatically *calibrated* (a 0.9 doesn't necessarily mean
right 90% of the time). Check before you rely on the numbers:"""),
C('''bins = np.linspace(0.33, 1, 8); which = np.digitize(maxp, bins) - 1
conf, acc = [], []
for k in range(len(bins) - 1):
    sel = which == k
    if sel.sum() >= 5: conf.append(maxp[sel].mean()); acc.append((pred[sel] == yte[sel]).mean())
plt.figure(figsize=(3.6, 3.4)); plt.plot([0.33, 1], [0.33, 1], "k--", lw=1, label="perfect calibration")
plt.plot(conf, acc, "o-", label="our model"); plt.xlabel("stated confidence"); plt.ylabel("actual accuracy"); plt.legend(fontsize=8)
plt.title("Reliability diagram"); plt.tight_layout(); plt.show()'''),
M("""### 🧩 Exercise — recall from a confusion matrix

Using `Mx` (rows = true, columns = predicted), compute the recall for `fish` (class 0) yourself, without calling
`class_metrics`. Store it in `fish_recall`."""),
C('''fish_recall = None   # YOUR CODE

def check_recall(v):
    exp = Mx[0, 0] / Mx[0].sum()
    if v is None: print("Not yet: replace None."); return
    if abs(v - exp) < 1e-9: print(f"✅ Correct: {exp:.3f}")
    elif abs(v - Mx[0, 0] / Mx[:, 0].sum()) < 1e-9: print("❌ That's precision (you divided by the column). Recall divides by the row: all TRUE fish.")
    else: print(f"❌ Expected {exp:.3f}.")
check_recall(fish_recall)'''),
C('''#@title Solution (double-click to show)
fish_recall = Mx[0, 0] / Mx[0].sum()
check_recall(fish_recall)''', hidden=True),
M("""## Check yourself

<details><summary>1. Your model scores 0.98 with a random frame split and 0.81 with a session split. Which do you report?</summary>The session split. The 0.98 is inflated by near-duplicate frames shared between train and test.</details>
<details><summary>2. Why standardise features with training-set statistics only?</summary>Using test statistics leaks information about the test set into training.</details>
<details><summary>3. A fish detector has precision 0.95 and recall 0.40. Describe it in plain words.</summary>When it says fish it's usually right, but it misses most fish.</details>
<details><summary>4. What does raising the abstention threshold do to coverage and to accuracy on kept frames?</summary>Coverage falls (more frames go to review); accuracy on the frames still decided usually rises.</details>

**Next:** Module 05, *Robustness and honest reporting* — the same model under turbidity, low light and a new
habitat, and how to write that up so nobody over-reads it."""),
]
build(cells, "out/CV04_learning_to_classify.ipynb")
