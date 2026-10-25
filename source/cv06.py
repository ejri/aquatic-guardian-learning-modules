from nbcommon import *
cells = [
M("# Module 06 — Neural networks and CNNs\n\n" + HEADER.format(num="06", time="about 2.5 hours",
  skip="you have trained a CNN in PyTorch and fine-tuned a pretrained backbone.") + """
### By the end you can
1. Move image data into PyTorch's `(N, C, H, W)` layout and train with mini-batches.
2. Explain what a hidden layer adds, and what a convolutional layer adds on top.
3. Train a small CNN on CPU and compare it fairly with simpler models, by condition.
4. Look inside: first-layer filters, feature maps, and a saliency map.
5. Fine-tune a pretrained backbone on real data in Colab (the upgrade path in **Mentee 1, Task 6**).

**Runtime:** about 2 minutes on Colab's free CPU. A GPU (Runtime → Change runtime type) makes it faster but isn't needed.
"""),
C(SETUP, hidden=True),
C(FRAMES, hidden=True),
C('''import time, pandas as pd
try:
    import torch, torch.nn as nn, torch.nn.functional as Fnn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("PyTorch is not installed here. In Colab it is preinstalled; locally: pip install torch torchvision")
if HAVE_TORCH:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("PyTorch", torch.__version__, "| device:", device)'''),
M("""## 1. Data in PyTorch layout ▶

NumPy images are `(N, H, W, C)`. PyTorch convolutions expect `(N, C, H, W)`: channels first."""),
C('''Xtr, ytr, _, htr = make_clips(n_sessions_per_class=45, seed=10)
Xte, yte, _, hte = make_clips(n_sessions_per_class=15, seed=99)
HABITATS["harbour_far"] = dict(HABITATS["harbour"], dist=(2.5, 4.0)); HABITATS["reef_far"] = dict(HABITATS["reef"], dist=(5.0, 8.0))
Xfar, yfar, _, _ = make_clips(n_sessions_per_class=15, habitats=("reef_far", "harbour_far"), seed=7)
print("numpy:", Xtr.shape)
if HAVE_TORCH:
    def to_t(X, norm=False):
        X = np.asarray(X, dtype=np.float32)
        if norm:   # per-image standardisation, explained in section 3
            X = (X - X.mean((1, 2), keepdims=True)) / (X.std((1, 2), keepdims=True) + 1e-3)
        return torch.tensor(np.transpose(X, (0, 3, 1, 2)))
    Ttr, Tte = to_t(Xtr), to_t(Xte); Ytr, Yte = torch.tensor(ytr), torch.tensor(yte)
    print("torch:", tuple(Ttr.shape), "(N, C, H, W)")'''),
M("""## 2. Three models of increasing structure ▶

1. **Linear** on raw pixels: one weight per pixel per class. Same maths as Module 04's softmax, but no hand-made features.
2. **MLP**: one hidden layer with a nonlinearity (ReLU). It can learn combinations of pixels.
3. **CNN**: small filters slid across the image (Module 03's convolution, but the kernel values are *learned*),
   then pooling to shrink the image. It shares weights across positions, so a fish in the corner and a fish in
   the centre are detected by the same filters."""),
C('''if HAVE_TORCH:
    class Linear(nn.Module):
        def __init__(s): super().__init__(); s.f = nn.Linear(3 * HW * HW, 3)
        def forward(s, x): return s.f(x.flatten(1))
    class MLP(nn.Module):
        def __init__(s, h=128): super().__init__(); s.f = nn.Sequential(nn.Flatten(), nn.Linear(3 * HW * HW, h), nn.ReLU(), nn.Linear(h, 3))
        def forward(s, x): return s.f(x)
    class SmallCNN(nn.Module):
        def __init__(s, w=16):
            super().__init__()
            s.features = nn.Sequential(
                nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),          # 48 -> 24
                nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 24 -> 12
                nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))  # 12 -> 6
            s.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, 3))
        def forward(s, x): return s.head(s.features(x))

    def train(model, T, Y, epochs=8, bs=64, lr=2e-3, seed=0):
        torch.manual_seed(seed); model = model.to(device); opt = torch.optim.Adam(model.parameters(), lr)
        hist = []
        for ep in range(epochs):
            model.train(); perm = torch.randperm(len(T)); tot = 0.0
            for i in range(0, len(T), bs):
                idx = perm[i:i + bs]; xb, yb = T[idx].to(device), Y[idx].to(device)
                opt.zero_grad(); loss = Fnn.cross_entropy(model(xb), yb); loss.backward(); opt.step(); tot += loss.item() * len(idx)
            hist.append(tot / len(T))
        return model, hist

    @torch.no_grad()
    def accuracy(model, X, y):
        model.eval(); p = model(to_t(X, getattr(model, "norm", False)).to(device)).argmax(1).cpu().numpy(); return (p == y).mean()

    def by_condition(model):
        return {"clean": accuracy(model, Xte, yte),
                "turbid 0.3": accuracy(model, np.array([degrade(im, turbidity=0.3) for im in Xte]), yte),
                "low light 0.5": accuracy(model, np.array([degrade(im, low_light=0.5) for im in Xte]), yte),
                "blur k=3": accuracy(model, np.array([degrade(im, blur_k=3) for im in Xte]), yte),
                "farther (physics)": accuracy(model, Xfar, yfar)}

    results, curves, models = {}, {}, {}
    for name, M_ in [("linear", Linear()), ("MLP", MLP()), ("small CNN", SmallCNN())]:
        t0 = time.time(); m, h = train(M_, Ttr, Ytr)
        models[name], curves[name] = m, h
        results[name] = by_condition(m); results[name]["train s"] = round(time.time() - t0, 1)
        results[name]["params"] = sum(p.numel() for p in m.parameters())
    plt.figure(figsize=(5, 2.6))
    for k, h in curves.items(): plt.plot(h, label=k)
    plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend(); plt.show()
    display(pd.DataFrame(results).T.round(3))'''),
M("""✏️ **Interpret:** the MLP has roughly 40 times more parameters than the CNN, yet the CNN does better, because
its filters are shared across positions and suit images. Do any of the three beat Module 05's six hand-made
features (about 0.93 on clean frames)? If not, why might that be? (Hint: look at how dim and low-contrast the raw
frames are, and how much they differ in colour from one habitat to the next.)

## 3. Preprocessing: one line that changes a lot ▶

Our frames differ mostly in overall **colour cast and brightness** from habitat to habitat. Subtracting each
frame's own mean and dividing by its own standard deviation (per channel) removes that, so the network can
spend its capacity on shape. Same CNN, same epochs, one extra line."""),
C('''if HAVE_TORCH:
    t0 = time.time(); cnn_n, _ = train(SmallCNN(), to_t(Xtr, norm=True), Ytr); cnn_n.norm = True
    models["small CNN, normalised"] = cnn_n
    results["small CNN, normalised"] = by_condition(cnn_n)
    results["small CNN, normalised"]["train s"] = round(time.time() - t0, 1)
    results["small CNN, normalised"]["params"] = results["small CNN"]["params"]
    display(pd.DataFrame(results).T.round(3))'''),
M("""✏️ **Interpret, carefully.** Look at which columns improve and which don't. Then look at `degrade()` in the
setup cell: its turbidity step is (almost) `a * image + constant` for the whole frame. Per-image standardisation
cancels exactly that kind of change. Blur changes the *spatial* structure, which standardisation can't undo.

This is a result about the **test**, not only about the model: a robustness score measured with `degrade()` can
look excellent for a model that simply normalises away `degrade()`'s particular distortion. Compare the improvement
on the physically farther objects. **Always test with more than one degradation model.**

## 4. Look inside the CNN ▶

**First-layer filters** are 3×3 colour kernels. **Feature maps** show where each filter responds."""),
C('''if HAVE_TORCH:
    cnn = models["small CNN, normalised"].cpu()
    w = cnn.features[0].weight.detach().numpy()            # (16, 3, 3, 3)
    fig, ax = plt.subplots(2, 8, figsize=(10, 2.8))
    for i, a in enumerate(ax.ravel()):
        k = np.transpose(w[i], (1, 2, 0)); k = (k - k.min()) / (k.max() - k.min() + 1e-9)
        a.imshow(k, interpolation="nearest"); a.axis("off")
    fig.suptitle("16 learned first-layer filters (each 3x3, colour)", fontsize=9); plt.show()
    fish_frame = Xte[np.where(yte == 0)[0][5]]
    with torch.no_grad(): fmap = cnn.features[:2](to_t(fish_frame[None], norm=True))[0].numpy()
    fig, ax = plt.subplots(1, 7, figsize=(13, 2.1)); ax[0].imshow(fish_frame); ax[0].set_title("input"); ax[0].axis("off")
    for i, a in enumerate(ax[1:]): a.imshow(fmap[i], cmap="magma"); a.set_title(f"filter {i}"); a.axis("off")
    plt.show(); cnn.to(device)'''),
M("""## 5. What is the model looking at? A saliency map ▶

Take the gradient of the predicted class score with respect to the input pixels. Large values mean "changing this
pixel would change the decision most". If the bright region isn't on the animal, the model may be using a shortcut
(a background colour that happens to correlate with the class in your data)."""),
C('''if HAVE_TORCH:
    cnn = models["small CNN, normalised"].cpu().eval()
    picks = [np.where(yte == c)[0][7] for c in range(3)]
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5))
    for j, k in enumerate(picks):
        x = to_t(Xte[k][None], norm=True).requires_grad_(True); out = cnn(x); c = out.argmax(1).item()
        out[0, c].backward(); sal = x.grad.abs()[0].max(0).values.numpy()
        ax[0, j].imshow(Xte[k]); ax[0, j].set_title(f"true {CLASSES[yte[k]]}\\npred {CLASSES[c]}", fontsize=8); ax[0, j].axis("off")
        ax[1, j].imshow(sal, cmap="inferno"); ax[1, j].axis("off")
    plt.tight_layout(); plt.show(); cnn.to(device)'''),
M("""## 6. Augmentation on top ▶"""),
C('''if HAVE_TORCH:
    r = np.random.default_rng(3); aug = []
    for im in Xtr:
        kw = [dict(turbidity=0.35), dict(low_light=0.5), dict(turbidity=0.25, blur_k=3), dict(blur_k=3)][r.integers(4)]
        aug.append(degrade(im, **kw))
    Taug = torch.cat([to_t(Xtr, norm=True), to_t(np.array(aug), norm=True)]); Yaug = torch.cat([Ytr, Ytr])
    t0 = time.time(); cnn_aug, _ = train(SmallCNN(), Taug, Yaug); cnn_aug.norm = True
    key = "small CNN, normalised + augmentation"
    results[key] = by_condition(cnn_aug); results[key]["train s"] = round(time.time() - t0, 1)
    results[key]["params"] = results["small CNN"]["params"]
    display(pd.DataFrame(results).T.round(3))'''),
M("""✏️ **Interpret:** compare the four `degrade()` columns with the *farther (physics)* column. A score that looks
almost perfect under one degradation model can hide a large gap under another. Write both into any report, and
treat real turbid footage as the only test that settles it.

## 7. Size and latency ▶

The project's later phase runs models on a small device, so report parameters and time per image, measured the same
way for every model (Mentee 1, Task 9)."""),
C('''if HAVE_TORCH:
    x1 = to_t(Xte[:1]).to(device)
    for name, m in [("linear", models["linear"]), ("MLP", models["MLP"]), ("small CNN", models["small CNN, normalised"])]:
        m.eval()
        with torch.no_grad():
            for _ in range(5): m(x1)
            t0 = time.time()
            for _ in range(100): m(x1)
        print(f"{name:10s} {sum(p.numel() for p in m.parameters()):>8,d} params   {(time.time() - t0) * 10:.2f} ms/image on {device}")'''),
M("""## 8. Real data: fine-tuning a pretrained backbone (Colab)

A network pretrained on millions of everyday photos already knows edges, textures and shapes. Replacing its last
layer and training briefly on your data is **transfer learning**. This is the Mentee 1 Task 6 upgrade path.

Set `RUN_REAL_DATA = True` in Colab, arrange images as `data_dir/<class>/*.jpg` (for example, frames exported from
DeepFish or Brackish, mapped onto fish / non_fish / empty_frame, **split by clip**), and record the licence."""),
C('''RUN_REAL_DATA = False
DATA_DIR = "/content/aquatic_frames"     # data_dir/fish, data_dir/non_fish, data_dir/empty_frame

def finetune_mobilenet(data_dir, epochs=5, img=128, batch=32):
    from torchvision import datasets, transforms, models as tvm
    from torch.utils.data import DataLoader
    tf = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tr = datasets.ImageFolder(f"{data_dir}/train", tf); va = datasets.ImageFolder(f"{data_dir}/val", tf)   # split by clip beforehand
    net = tvm.mobilenet_v3_small(weights="DEFAULT")
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(tr.classes))
    for p in net.features.parameters(): p.requires_grad = False          # train the new head first
    net = net.to(device); opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], 1e-3)
    for ep in range(epochs):
        net.train()
        for xb, yb in DataLoader(tr, batch, shuffle=True):
            opt.zero_grad(); Fnn.cross_entropy(net(xb.to(device)), yb.to(device)).backward(); opt.step()
    net.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in DataLoader(va, batch):
            correct += (net(xb.to(device)).argmax(1).cpu() == yb).sum().item(); total += len(yb)
    print(f"validation accuracy: {correct / total:.3f}  classes: {tr.classes}")
    return net

if HAVE_TORCH and RUN_REAL_DATA:
    real_model = finetune_mobilenet(DATA_DIR)
else:
    print("Skipped (RUN_REAL_DATA = False). In Colab: prepare DATA_DIR/train and DATA_DIR/val, then set it to True.")'''),
M("""### 🧩 Exercise — shapes through the network

Our CNN takes a 48×48 image. Each `Conv2d(..., 3, padding=1)` keeps the size; each `MaxPool2d(2)` halves it.
What is the spatial size (height = width) of the output of `cnn.features`? Work it out, then store it in
`spatial_size`. A checker runs a real tensor through to confirm."""),
C('''spatial_size = None   # YOUR ANSWER (an integer)

def check_size(v):
    if v is None: print("Not yet: replace None with an integer."); return
    if HAVE_TORCH:
        with torch.no_grad(): s = models["small CNN"].cpu().features(torch.zeros(1, 3, HW, HW)).shape[-1]
        models["small CNN"].to(device)
    else: s = HW // 8
    print("✅ Correct:" if v == s else "❌ Not quite: three halvings of 48.", f"actual size = {s}")
check_size(spatial_size)'''),
C('''#@title Solution (double-click to show)
spatial_size = 48 // 2 // 2 // 2
check_size(spatial_size)''', hidden=True),
M("""## Check yourself

<details><summary>1. Why does the CNN need far fewer parameters than a linear model on raw pixels?</summary>Its small filters are reused at every position (weight sharing), instead of one weight per pixel per class.</details>
<details><summary>2. Your saliency map lights up the background, not the fish. What might be wrong?</summary>A shortcut: some background property correlates with the label in your training data. Check class balance per habitat and per clip.</details>
<details><summary>3. When fine-tuning a pretrained network, why freeze the early layers first?</summary>They hold general features (edges, textures). Training only the new head first is faster and less likely to overfit a small dataset.</details>
<details><summary>4. Your model scores 0.97 under degrade(turbidity=0.3). Is it robust to turbid water?</summary>Not necessarily. It may only have learned to cancel that function's particular distortion. Test with a different degradation model (for example the physics model) and, ideally, real turbid footage.</details>

**Next:** Module 07, *Detection, measurement and calibration* — from "is there a fish?" to "where, how big, and how heavy?"."""),
]
build(cells, "out/CV06_neural_networks_and_cnns.ipynb")
