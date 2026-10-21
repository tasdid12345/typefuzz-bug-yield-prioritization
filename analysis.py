"""Seeded fuzzing-campaign simulation: type-coverage queue (baseline) vs a
class-balanced LightGBM-prioritised queue (improved), same run budget, same test campaigns."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED = 42
RUN_SEEDS = [42, 43, 44]          # seeds for randomized campaigns and LightGBM
N_CAMPAIGNS = 40                  # simulated campaigns (train + test)
N_STATES = 1500                   # type states in each simulated universe
N_FEAT = 12                       # length of a type-state feature vector
HIDE_RATE = 0.03                  # fraction of type states that hide a bug
TRIGGER_P = 0.05                  # per-run bug trigger probability on a hiding state
DISCOVER_P = 0.2                  # chance a run reveals a new type state
BUDGET = 2000                     # fixed run budget per campaign (same for both)
W = np.array([1.0, -0.8, 0.6, 0.5, -0.4, 0.9] + [0.0] * (N_FEAT - 6))  # hidden signal
OUT = "results.json"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def make_universe(c):
    """Type-state signatures and hidden bug-hiding flags for simulated campaign c."""
    rng = np.random.default_rng(1000 + c)
    X = rng.normal(size=(N_STATES, N_FEAT))
    s = X @ W + rng.normal(scale=0.5, size=N_STATES)      # noisy bug propensity
    hide = s >= np.quantile(s, 1 - HIDE_RATE)             # top 3% hide a bug
    return X, hide


def run_campaign(hide, scheduler, scores, rng):
    """One campaign with a fixed run budget.
    'fifo' (baseline): round-robin over the queue in first-come order; every new type
    state gets equal priority. 'model' (improved): run the queued type state with the
    highest predicted bug yield, discounted by runs already spent on it."""
    n = len(hide)
    order = rng.permutation(n)                  # order in which type states are revealed
    queue, runs = [], np.zeros(n, int)
    bug = np.zeros(n, bool)                     # one distinct seeded bug per hiding state
    n_seen, first_bug, rr = 0, BUDGET, 0
    for b in range(BUDGET):
        # reveal a new type state on the first run, then with probability DISCOVER_P
        if n_seen < n and (not queue or rng.random() < DISCOVER_P):
            queue.append(order[n_seen])
            n_seen += 1
        q = np.asarray(queue)
        if scheduler == "fifo":
            i = q[rr % len(q)]
            rr += 1
        else:
            i = q[np.argmax(scores[q] / (1 + runs[q]))]
        runs[i] += 1
        if hide[i] and not bug[i] and rng.random() < TRIGGER_P:
            bug[i] = True
            first_bug = min(first_bug, b + 1)
    return {"bugs": int(bug.sum()), "first": first_bug, "covered": len(queue),
            "bug": bug, "seen": np.asarray(queue)}


def build_train(camps):
    """Training table: type-state signatures and bug labels from earlier baseline-style campaigns."""
    Xs, ys = [], []
    for c in camps:
        X, hide = make_universe(c)
        r = run_campaign(hide, "fifo", None, np.random.default_rng(7000 + c))
        Xs.append(X[r["seen"]])
        ys.append(r["bug"][r["seen"]])
    return np.vstack(Xs), np.concatenate(ys).astype(int)


def tune(X, y):
    """Small class-balanced LightGBM grid, selected by 3-fold CV on training rows only."""
    grid = {"num_leaves": [7, 15], "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 200], "min_child_samples": [10, 30]}
    base = lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    search = GridSearchCV(base, grid, scoring="average_precision", cv=cv).fit(X, y)
    return search.best_params_, search.best_score_


def main():
    camps = np.arange(N_CAMPAIGNS)
    tr, te = train_test_split(camps, test_size=0.3, random_state=SEED)   # split once
    log(f"campaigns: {len(tr)} train / {len(te)} test")
    X_tr, y_tr = build_train(tr)
    log(f"training rows {len(y_tr)}, bug-yielding rows {int(y_tr.sum())}")
    if y_tr.sum() < 10:
        log("not enough bug-yielding training rows; cannot produce meaningful metrics")
        sys.exit(1)
    params, cv_ap = tune(X_tr, y_tr)
    log(f"CV average precision {cv_ap:.3f} with {params}")

    res = {k: {"bugs": [], "first": [], "cov": []} for k in ["baseline", "improved"]}
    aucs = []
    for s in RUN_SEEDS:
        model = lgb.LGBMClassifier(**params, class_weight="balanced",
                                   random_state=s, verbose=-1).fit(X_tr, y_tr)
        acc = {k: {"bugs": [], "first": [], "cov": []} for k in res}
        for c in te:
            X, hide = make_universe(c)
            scores = model.predict_proba(X)[:, 1]
            aucs.append(roc_auc_score(hide, scores))
            # same universe and same campaign RNG seed for both schedulers
            for name, sched, sc in [("baseline", "fifo", None), ("improved", "model", scores)]:
                r = run_campaign(hide, sched, sc, np.random.default_rng(s * 100000 + c))
                acc[name]["bugs"].append(r["bugs"])
                acc[name]["first"].append(r["first"])
                acc[name]["cov"].append(r["covered"])
        for name in res:   # per-seed means over test campaigns
            for k in res[name]:
                res[name][k].append(float(np.mean(acc[name][k])))
        log(f"seed {s}: baseline bugs {res['baseline']['bugs'][-1]:.2f}, "
            f"improved bugs {res['improved']['bugs'][-1]:.2f}")

    b = np.array(res["baseline"]["bugs"])
    i = np.array(res["improved"]["bugs"])
    metrics = {
        "baseline_score": float(b.mean()), "improved_score": float(i.mean()),
        "improvement": float(i.mean() - b.mean()),
        "baseline_score_std": float(b.std(ddof=1)), "improved_score_std": float(i.std(ddof=1)),
        "baseline_runs_to_first_bug": float(np.mean(res["baseline"]["first"])),
        "improved_runs_to_first_bug": float(np.mean(res["improved"]["first"])),
        "baseline_states_covered": float(np.mean(res["baseline"]["cov"])),
        "improved_states_covered": float(np.mean(res["improved"]["cov"])),
        "improved_model_test_auc": float(np.mean(aucs)),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        log("non-finite metric; failing")
        sys.exit(1)
    for k, v in metrics.items():
        log(f"{k}: {v:.3f}")

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["baseline", "improved"], [b.mean(), i.mean()],
           yerr=[b.std(ddof=1), i.std(ddof=1)], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("distinct seeded bugs found (mean per campaign)")
    fig.tight_layout()
    fig.savefig("figures/bugs_found.png", dpi=120)

    out = {
        "question": "Can a class-balanced gradient-boosting model that ranks type states by "
                    "predicted bug yield find more seeded bugs within a fixed run budget than a "
                    "type-coverage queue that treats every new type state equally?",
        "method": "Seeded simulation of fuzzing campaigns. Type states have feature vectors; a "
                  "small fraction hide bugs with low per-run trigger probability. A class-balanced "
                  "LightGBM trained on earlier campaigns scores type states; the same queue and "
                  "run budget are used for both schedulers. Campaigns are split once into train "
                  "and test; tuning uses 3-fold CV on training rows only.",
        "dataset": {
            "name": "Seeded fuzzing-campaign simulation (numpy, no download)",
            "source_url": "https://numpy.org/",
            "n_rows": N_CAMPAIGNS * N_STATES,
            "n_train": int(len(tr) * N_STATES),
            "n_test": int(len(te) * N_STATES),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "distinct seeded bugs found within fixed run budget (mean per test campaign)",
        "higher_is_better": True,
        "baseline": "Type-coverage-guided queue: round-robin over type states in first-come order, "
                    "every new type state has equal priority.",
        "improved": "Same queue and budget, but each run goes to the queued type state with the "
                    "highest class-balanced LightGBM bug-yield score (discounted by runs spent).",
        "metrics": metrics,
        "notes": "Simulation only; the hidden bug signal is linear in the features, which favours "
                 "the tree model, so gains may be optimistic. Training labels come from baseline-style "
                 "campaigns. Campaign-level split (not row-stratified) to avoid leakage. Bugs are "
                 "rare, so the metric is noisy; standard deviations are across three seeds.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
