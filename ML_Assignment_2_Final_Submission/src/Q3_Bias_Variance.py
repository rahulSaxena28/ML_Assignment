"""Q3 — Bias and variance measured over 30 resamples
Generated from ML_Assignment_2_Solution.ipynb.
Run from a Python environment with numpy, pandas, matplotlib, and scikit-learn installed.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 26006
rng = np.random.default_rng(SEED)

np.random.seed(SEED)
print("Seed:", SEED)

# If you already have `olivetti_faces` loaded in your notebook, this works as-is.
olivetti_faces = fetch_olivetti_faces(shuffle=False)
X_all = olivetti_faces.data.astype(np.float32)
y_all = olivetti_faces.target.astype(int)

print("X shape:", X_all.shape)
print("Number of identities:", len(np.unique(y_all)))

split_file = "olivetti_split_seed_26006.npz"

if os.path.exists(split_file):
    saved = np.load(split_file)
    train_idx = saved["train_idx"]
    val_idx = saved["val_idx"]
    test_idx = saved["test_idx"]
else:
    split_rng = np.random.default_rng(SEED)
    train_idx, val_idx, test_idx = [], [], []
    for identity in range(40):
        indices = np.where(y_all == identity)[0]
        split_rng.shuffle(indices)
        train_idx.extend(indices[:6])
        val_idx.extend(indices[6:8])
        test_idx.extend(indices[8:10])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

Xtr, ytr = X_all[train_idx], y_all[train_idx]
Xval, yval = X_all[val_idx], y_all[val_idx]
Xte, yte = X_all[test_idx], y_all[test_idx]

print("Train:", Xtr.shape, "Validation:", Xval.shape, "Test:", Xte.shape)
assert len(train_idx) == 240 and len(val_idx) == 80 and len(test_idx) == 80

def mask_faces(X):
    images = X.reshape(-1, 64, 64).copy()
    images[:, 24:40, 24:40] = 0.0
    return images.reshape(-1, 4096)

Xte_masked = mask_faces(Xte)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(Xte[0].reshape(64, 64), cmap="gray")
axes[0].set_title("Clean test face")
axes[1].imshow(Xte_masked[0].reshape(64, 64), cmap="gray")
axes[1].set_title("Masked test face")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

tree_results = []

for depth in [2, 4, 8, None]:
    for leaf in [1, 2, 4]:
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=SEED
        )
        start = time.time()
        model.fit(Xtr, ytr)
        elapsed = time.time() - start
        train_acc = accuracy_score(ytr, model.predict(Xtr))
        val_acc = accuracy_score(yval, model.predict(Xval))
        tree_results.append({
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "fit_seconds": elapsed
        })

tree_results = pd.DataFrame(tree_results).sort_values(
    ["validation_accuracy", "max_depth", "min_samples_leaf"],
    ascending=[False, True, True],
    na_position="last"
)
display(tree_results)

best_row = tree_results.iloc[0]
BEST_DEPTH = best_row["max_depth"]
BEST_LEAF = int(best_row["min_samples_leaf"])
print("Frozen setting:", BEST_DEPTH, BEST_LEAF)

tree = DecisionTreeClassifier(
    max_depth=None if pd.isna(BEST_DEPTH) else int(BEST_DEPTH),
    min_samples_leaf=BEST_LEAF,
    random_state=SEED
)
tree.fit(Xtr, ytr)

tree_train_acc = accuracy_score(ytr, tree.predict(Xtr))
tree_val_acc = accuracy_score(yval, tree.predict(Xval))
tree_test_acc = accuracy_score(yte, tree.predict(Xte))
print(f"Train accuracy: {tree_train_acc:.4f}")
print(f"Validation accuracy: {tree_val_acc:.4f}")
print(f"Clean test accuracy: {tree_test_acc:.4f}")

def fit_bagging(X, y, n_trees, max_depth, min_samples_leaf, seed=SEED):
    local_rng = np.random.default_rng(seed)
    models = []
    boot_indices = []
    oob_indices = []
    n = len(y)

    for b in range(n_trees):
        boot = local_rng.integers(0, n, size=n)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[boot] = True
        oob = np.where(~in_bag)[0]

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed + b
        )
        model.fit(X[boot], y[boot])
        models.append(model)
        boot_indices.append(boot)
        oob_indices.append(oob)

    return models, boot_indices, oob_indices

def ensemble_predictions(models, X):
    return np.array([model.predict(X) for model in models]).T

def majority_vote(predictions):
    # predictions shape: (n_examples, n_trees)
    out = []
    for row in predictions:
        labels, counts = np.unique(row, return_counts=True)
        out.append(labels[np.argmax(counts)])
    return np.array(out)

def vote_statistics(predictions, classes):
    # Predictions: examples x trees. Return winning fraction, runner-up fraction,
    # and winning-minus-runner-up vote margin for each example.
    fractions = np.zeros((predictions.shape[0], len(classes)))
    class_to_col = {c: j for j, c in enumerate(classes)}
    for i, row in enumerate(predictions):
        for label in row:
            fractions[i, class_to_col[label]] += 1
        fractions[i] /= len(row)
    sorted_votes = np.sort(fractions, axis=1)
    margin = sorted_votes[:, -1] - sorted_votes[:, -2]
    confidence = sorted_votes[:, -1]
    return confidence, margin, fractions

B_VALUES = [1, 5, 10, 25, 50, 100, 200]
bag_models_by_B = {}
bag_rows = []

for B in B_VALUES:
    start = time.time()
    models, boots, oobs = fit_bagging(
        Xtr, ytr, B, BEST_DEPTH, BEST_LEAF, seed=SEED + B
    )
    pred_train = ensemble_predictions(models, Xtr)
    pred_val = ensemble_predictions(models, Xval)
    pred_test = ensemble_predictions(models, Xte)

    train_acc = accuracy_score(ytr, majority_vote(pred_train))
    val_acc = accuracy_score(yval, majority_vote(pred_val))
    test_acc = accuracy_score(yte, majority_vote(pred_test))

    oob_true, oob_pred = [], []
    for i in range(len(ytr)):
        votes = []
        for model, oob in zip(models, oobs):
            if i in oob:
                votes.append(model.predict(Xtr[i:i+1])[0])
        if votes:
            oob_true.append(ytr[i])
            oob_pred.append(majority_vote(np.array(votes).reshape(1, -1))[0])

    oob_acc = accuracy_score(oob_true, oob_pred) if oob_true else np.nan
    bag_models_by_B[B] = models
    bag_rows.append({
        "B": B,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "clean_test_accuracy": test_acc,
        "oob_accuracy": oob_acc,
        "average_distinct_examples_per_bootstrap": np.mean([len(np.unique(x)) for x in boots]),
        "training_seconds": time.time() - start
    })
    print("Finished B =", B)

bag_results = pd.DataFrame(bag_rows)
display(bag_results)

def pairwise_disagreement(predictions):
    n_trees = predictions.shape[1]
    if n_trees < 2:
        return 0.0
    values = []
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            values.append(np.mean(predictions[:, i] != predictions[:, j]))
    return float(np.mean(values))

disagreement_rows = []
for B in B_VALUES:
    models = bag_models_by_B[B]
    pred_val = ensemble_predictions(models, Xval)
    confidence, margin, _ = vote_statistics(pred_val, np.unique(ytr))
    disagreement_rows.append({
        "B": B,
        "validation_accuracy": accuracy_score(yval, majority_vote(pred_val)),
        "pairwise_disagreement": pairwise_disagreement(pred_val),
        "mean_vote_margin": float(np.mean(margin)),
        "mean_winning_vote_fraction": float(np.mean(confidence))
    })

disagreement_results = pd.DataFrame(disagreement_rows)
display(disagreement_results)

best_B = int(bag_results.sort_values(
    ["validation_accuracy", "B"], ascending=[False, True]
).iloc[0]["B"])
final_bag = bag_models_by_B[best_B]
print("Selected B:", best_B)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].plot(bag_results["B"], bag_results["oob_accuracy"], marker="o")
axes[0].set_title("OOB accuracy")
axes[1].plot(disagreement_results["B"], disagreement_results["pairwise_disagreement"], marker="o")
axes[1].set_title("Pairwise disagreement")
axes[2].plot(disagreement_results["B"], disagreement_results["mean_vote_margin"], marker="o")
axes[2].set_title("Mean vote margin")
for ax in axes:
    ax.set_xlabel("Number of trees")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

tree_clean_pred = tree.predict(Xte)
tree_mask_pred = tree.predict(Xte_masked)

bag_clean_votes = ensemble_predictions(final_bag, Xte)
bag_mask_votes = ensemble_predictions(final_bag, Xte_masked)
bag_clean_pred = majority_vote(bag_clean_votes)
bag_mask_pred = majority_vote(bag_mask_votes)

bag_clean_conf, bag_clean_margin, _ = vote_statistics(bag_clean_votes, np.unique(ytr))
bag_mask_conf, bag_mask_margin, _ = vote_statistics(bag_mask_votes, np.unique(ytr))

q1d_results = pd.DataFrame([
    {"model": "Single tree", "clean_accuracy": accuracy_score(yte, tree_clean_pred),
     "masked_accuracy": accuracy_score(yte, tree_mask_pred)},
    {"model": f"Bagging B={best_B}", "clean_accuracy": accuracy_score(yte, bag_clean_pred),
     "masked_accuracy": accuracy_score(yte, bag_mask_pred)}
])
display(q1d_results)

corrected = np.where((tree_mask_pred != yte) & (bag_mask_pred == yte))[0]
both_wrong = np.where((tree_mask_pred != yte) & (bag_mask_pred != yte))[0]
print("Tree wrong, bagging correct on masked set:", len(corrected))
print("Both wrong on masked set:", len(both_wrong))

def show_cases(indices, title, max_cases=6):
    indices = indices[:max_cases]
    if len(indices) == 0:
        print("No examples for:", title)
        return
    fig, axes = plt.subplots(len(indices), 1, figsize=(3, 2.3 * len(indices)))
    if len(indices) == 1:
        axes = [axes]
    for ax, i in zip(axes, indices):
        ax.imshow(Xte_masked[i].reshape(64, 64), cmap="gray")
        ax.set_title(
            f"true={yte[i]}, tree={tree_mask_pred[i]}, bag={bag_mask_pred[i]}, "
            f"bag confidence={bag_mask_conf[i]:.2f}"
        )
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

show_cases(corrected, "Masked cases corrected by bagging")
show_cases(both_wrong, "Masked cases where both models fail")

class SAMME:
    def __init__(self, n_estimators=50, max_depth=1, min_samples_leaf=1, random_state=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.models = []
        self.alphas = []
        self.classes_ = None
        self.history_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n = len(y)
        weights = np.ones(n) / n
        self.models = []
        self.alphas = []
        self.history_ = []

        for b in range(self.n_estimators):
            model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + b
            )
            model.fit(X, y, sample_weight=weights)
            pred = model.predict(X)
            incorrect = pred != y
            error = float(np.sum(weights * incorrect))

            # Stop if the weak learner is no better than random multiclass prediction.
            if error >= 1.0 - (1.0 / K):
                if b == 0:
                    raise ValueError("First weak learner is no better than random; change depth or seed.")
                break

            error = max(error, 1e-12)
            alpha = np.log((1.0 - error) / error) + np.log(K - 1)

            self.models.append(model)
            self.alphas.append(float(alpha))

            weights *= np.exp(alpha * incorrect)
            weights /= weights.sum()

            self.history_.append({
                "round": b + 1,
                "weighted_error": error,
                "alpha": alpha,
                "max_sample_weight": weights.max(),
                "total_sample_weight": weights.sum(),
                "training_error": np.mean(pred != y)
            })

            if error == 0:
                break

        self.alphas = np.array(self.alphas)
        return self

    def decision_function(self, X):
        scores = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_to_col = {c: j for j, c in enumerate(self.classes_)}
        for model, alpha in zip(self.models, self.alphas):
            pred = model.predict(X)
            for i, label in enumerate(pred):
                scores[i, class_to_col[label]] += alpha
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_confidence(self, X):
        scores = self.decision_function(X)
        totals = scores.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        normalized = scores / totals
        return normalized.max(axis=1), normalized

# Small reproducible toy example with 10 samples and 3 classes
X_toy = np.array([
    [0.0, 0.0], [0.1, 0.0], [0.2, 0.1], [0.3, 0.2],
    [0.8, 0.8], [0.9, 0.7], [1.0, 0.9],
    [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]
])
y_toy = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
w_toy = np.ones(len(y_toy)) / len(y_toy)

toy_learner = DecisionTreeClassifier(max_depth=1, random_state=SEED)
toy_learner.fit(X_toy, y_toy, sample_weight=w_toy)
toy_pred = toy_learner.predict(X_toy)
toy_error = np.sum(w_toy * (toy_pred != y_toy))
K_toy = len(np.unique(y_toy))
toy_alpha = np.log((1 - toy_error) / toy_error) + np.log(K_toy - 1)
w_toy_new = w_toy * np.exp(toy_alpha * (toy_pred != y_toy))
w_toy_new = w_toy_new / w_toy_new.sum()

print("Initial weights:", np.round(w_toy, 4))
print("True labels:     ", y_toy)
print("Predictions:     ", toy_pred)
print("Weighted error:", round(float(toy_error), 4))
print("Alpha:", round(float(toy_alpha), 4))
print("Updated weights:", np.round(w_toy_new, 4))
print("Updated sum:", w_toy_new.sum())

BOOST_DEPTHS = [1, 2, 3, 4]
BOOST_ROUNDS = [10, 25, 50, 100]
boost_results = []
boost_models = {}

for depth in BOOST_DEPTHS:
    for rounds in BOOST_ROUNDS:
        start = time.time()
        model = SAMME(
            n_estimators=rounds,
            max_depth=depth,
            min_samples_leaf=1,
            random_state=SEED
        )
        try:
            model.fit(Xtr, ytr)
            train_acc = accuracy_score(ytr, model.predict(Xtr))
            val_acc = accuracy_score(yval, model.predict(Xval))
            boost_models[(depth, rounds)] = model
            boost_results.append({
                "depth": depth, "requested_rounds": rounds,
                "accepted_rounds": len(model.models),
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "training_seconds": time.time() - start
            })
            print("Finished depth, rounds:", depth, rounds)
        except ValueError as e:
            print("Skipped:", depth, rounds, e)

boost_results = pd.DataFrame(boost_results)
display(boost_results.sort_values("validation_accuracy", ascending=False))

best_boost_row = boost_results.sort_values(
    ["validation_accuracy", "requested_rounds"], ascending=[False, True]
).iloc[0]
BEST_BOOST_DEPTH = int(best_boost_row["depth"])
BEST_BOOST_ROUNDS = int(best_boost_row["requested_rounds"])
boost = boost_models[(BEST_BOOST_DEPTH, BEST_BOOST_ROUNDS)]
print("Selected SAMME model:", BEST_BOOST_DEPTH, BEST_BOOST_ROUNDS)
display(pd.DataFrame(boost.history_).head())

boost_history = pd.DataFrame(boost.history_)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes[0, 0].plot(boost_history["round"], boost_history["weighted_error"], marker=".")
axes[0, 0].set_title("Weighted learner error")
axes[0, 1].plot(boost_history["round"], boost_history["alpha"], marker=".")
axes[0, 1].set_title("Learner weight alpha")
axes[1, 0].plot(boost_history["round"], boost_history["max_sample_weight"], marker=".")
axes[1, 0].set_title("Maximum sample weight")
axes[1, 1].plot(boost_history["round"], boost_history["training_error"], marker=".")
axes[1, 1].set_title("Weak learner training error")
for ax in axes.flat:
    ax.set_xlabel("Round")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

hardest = int(np.argmax(boost_history["max_sample_weight"].to_numpy()))
print("Round with largest maximum sample weight:", hardest + 1)
print("Largest sample weight:", boost_history.iloc[hardest]["max_sample_weight"])

comparison_rows = []
for B in [25, 50, 100]:
    start = time.time()
    models, _, _ = fit_bagging(
        Xtr, ytr, B, BEST_DEPTH, BEST_LEAF, seed=SEED + 1000 + B
    )
    bag_time = time.time() - start
    bag_train = accuracy_score(ytr, majority_vote(ensemble_predictions(models, Xtr)))
    bag_val = accuracy_score(yval, majority_vote(ensemble_predictions(models, Xval)))
    bag_test = accuracy_score(yte, majority_vote(ensemble_predictions(models, Xte)))
    comparison_rows.append({
        "method": "Bagging", "learners": B,
        "train_accuracy": bag_train, "validation_accuracy": bag_val,
        "clean_test_accuracy": bag_test, "training_seconds": bag_time
    })

    start = time.time()
    boost_model = SAMME(
        n_estimators=B, max_depth=BEST_BOOST_DEPTH,
        min_samples_leaf=1, random_state=SEED
    ).fit(Xtr, ytr)
    boost_time = time.time() - start
    comparison_rows.append({
        "method": "SAMME", "learners": B,
        "train_accuracy": accuracy_score(ytr, boost_model.predict(Xtr)),
        "validation_accuracy": accuracy_score(yval, boost_model.predict(Xval)),
        "clean_test_accuracy": accuracy_score(yte, boost_model.predict(Xte)),
        "training_seconds": boost_time
    })

comparison_results = pd.DataFrame(comparison_rows)
display(comparison_results)

# Single-tree leaf confidence
def tree_leaf_confidence(model, X):
    leaf_ids = model.apply(X)
    leaf_conf = {}
    for leaf in np.unique(leaf_ids):
        idx = model.apply(Xtr) == leaf
        if np.any(idx):
            _, counts = np.unique(ytr[idx], return_counts=True)
            leaf_conf[leaf] = counts.max() / counts.sum()
        else:
            leaf_conf[leaf] = 0.0
    return np.array([leaf_conf[leaf] for leaf in leaf_ids])

single_clean_conf = tree_leaf_confidence(tree, Xte)
single_mask_conf = tree_leaf_confidence(tree, Xte_masked)
boost_clean_conf, _ = boost.predict_confidence(Xte)
boost_mask_conf, _ = boost.predict_confidence(Xte_masked)

model_test_rows = [
    {"model": "Single tree", "clean_accuracy": accuracy_score(yte, tree_clean_pred),
     "masked_accuracy": accuracy_score(yte, tree_mask_pred),
     "clean_mean_confidence": np.mean(single_clean_conf),
     "masked_mean_confidence": np.mean(single_mask_conf)},
    {"model": f"Bagging B={best_B}", "clean_accuracy": accuracy_score(yte, bag_clean_pred),
     "masked_accuracy": accuracy_score(yte, bag_mask_pred),
     "clean_mean_confidence": np.mean(bag_clean_conf),
     "masked_mean_confidence": np.mean(bag_mask_conf)},
    {"model": "SAMME", "clean_accuracy": accuracy_score(yte, boost.predict(Xte)),
     "masked_accuracy": accuracy_score(yte, boost.predict(Xte_masked)),
     "clean_mean_confidence": np.mean(boost_clean_conf),
     "masked_mean_confidence": np.mean(boost_mask_conf)}
]
model_test_results = pd.DataFrame(model_test_rows)
display(model_test_results)

# Face where bagging and boosting disagree, with sample-weight history available for SAMME.
bag_mask_pred_final = bag_mask_pred
boost_mask_pred = boost.predict(Xte_masked)
disagree = np.where(bag_mask_pred_final != boost_mask_pred)[0]
print("Bagging vs boosting disagreement count on masked test:", len(disagree))
if len(disagree):
    i = int(disagree[0])
    plt.figure(figsize=(3, 3))
    plt.imshow(Xte_masked[i].reshape(64, 64), cmap="gray")
    plt.title(f"true={yte[i]}, bag={bag_mask_pred_final[i]}, boost={boost_mask_pred[i]}")
    plt.axis("off")
    plt.show()
    print("Bagging confidence:", bag_mask_conf[i])
    print("Boosting confidence:", boost_mask_conf[i])

N_REPEATS = 30
CLASSES = np.unique(ytr)
class_to_col = {c: j for j, c in enumerate(CLASSES)}

def resample_training_by_identity(X, y, seed):
    local_rng = np.random.default_rng(seed)
    selected = []
    for identity in CLASSES:
        idx = np.where(y == identity)[0]
        selected.extend(local_rng.choice(idx, size=len(idx), replace=True))
    selected = np.array(selected)
    return X[selected], y[selected]

def one_hot_predictions(pred, classes=CLASSES):
    out = np.zeros((len(pred), len(classes)), dtype=float)
    for i, label in enumerate(pred):
        out[i, class_to_col[label]] = 1.0
    return out

def summarize_prediction_runs(prediction_runs, y_true):
    # prediction_runs shape: repeats x examples x classes
    mean_prob = prediction_runs.mean(axis=0)
    true_one_hot = one_hot_predictions(y_true)
    bias2 = np.mean(np.sum((mean_prob - true_one_hot) ** 2, axis=1))
    variance = np.mean(np.sum(np.var(prediction_runs, axis=0), axis=1))
    accuracies = []
    for run_probs in prediction_runs:
        pred = CLASSES[np.argmax(run_probs, axis=1)]
        accuracies.append(accuracy_score(y_true, pred))
    return {
        "bias_squared": float(bias2),
        "variance": float(variance),
        "mean_test_accuracy": float(np.mean(accuracies)),
        "std_test_accuracy": float(np.std(accuracies, ddof=1)),
        "accuracies": accuracies
    }

single_runs = []
bag_runs = []
boost_runs = []
repeat_times = []

for r in range(N_REPEATS):
    Xr, yr = resample_training_by_identity(Xtr, ytr, SEED + 5000 + r)

    start = time.time()
    single_model = DecisionTreeClassifier(
        max_depth=None if pd.isna(BEST_DEPTH) else BEST_DEPTH,
        min_samples_leaf=BEST_LEAF,
        random_state=SEED + r
    ).fit(Xr, yr)
    single_runs.append(one_hot_predictions(single_model.predict(Xte)))
    repeat_times.append(time.time() - start)

    start = time.time()
    bag_models, _, _ = fit_bagging(
        Xr, yr, 100, BEST_DEPTH, BEST_LEAF, seed=SEED + 6000 + r
    )
    bag_pred = majority_vote(ensemble_predictions(bag_models, Xte))
    bag_runs.append(one_hot_predictions(bag_pred))
    repeat_times.append(time.time() - start)

    start = time.time()
    boost_model = SAMME(
        n_estimators=BEST_BOOST_ROUNDS,
        max_depth=BEST_BOOST_DEPTH,
        min_samples_leaf=1,
        random_state=SEED + 7000 + r
    ).fit(Xr, yr)
    boost_runs.append(one_hot_predictions(boost_model.predict(Xte)))
    repeat_times.append(time.time() - start)

    print(f"Finished resample {r + 1}/{N_REPEATS}")

single_runs = np.array(single_runs)
bag_runs = np.array(bag_runs)
boost_runs = np.array(boost_runs)

bias_variance_rows = []
for name, runs in [
    ("Single tree", single_runs),
    ("Bagging (100 trees)", bag_runs),
    ("SAMME boosting", boost_runs)
]:
    result = summarize_prediction_runs(runs, yte)
    bias_variance_rows.append({
        "model": name,
        "bias_squared": result["bias_squared"],
        "variance": result["variance"],
        "mean_test_accuracy": result["mean_test_accuracy"],
        "std_test_accuracy": result["std_test_accuracy"]
    })

bias_variance_results = pd.DataFrame(bias_variance_rows)
display(bias_variance_results)

def run_depth_bias_variance(depth, n_repeats=30, n_bag=100):
    tree_runs = []
    bagged_runs = []
    for r in range(n_repeats):
        Xr, yr = resample_training_by_identity(Xtr, ytr, SEED + 9000 + r)
        model = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=BEST_LEAF, random_state=SEED + r
        ).fit(Xr, yr)
        tree_runs.append(one_hot_predictions(model.predict(Xte)))

        models, _, _ = fit_bagging(
            Xr, yr, n_bag, depth, BEST_LEAF, seed=SEED + 10000 + r
        )
        pred = majority_vote(ensemble_predictions(models, Xte))
        bagged_runs.append(one_hot_predictions(pred))

    tree_summary = summarize_prediction_runs(np.array(tree_runs), yte)
    bag_summary = summarize_prediction_runs(np.array(bagged_runs), yte)
    return tree_summary, bag_summary

DEPTHS_TO_COMPARE = [2, BEST_DEPTH, 16]
depth_rows = []
for depth in DEPTHS_TO_COMPARE:
    single_summary, bag_summary = run_depth_bias_variance(depth)
    depth_rows.append({
        "depth": depth, "model": "Single tree",
        "bias_squared": single_summary["bias_squared"],
        "variance": single_summary["variance"],
        "mean_test_accuracy": single_summary["mean_test_accuracy"],
        "std_test_accuracy": single_summary["std_test_accuracy"]
    })
    depth_rows.append({
        "depth": depth, "model": "Bagging (100 trees)",
        "bias_squared": bag_summary["bias_squared"],
        "variance": bag_summary["variance"],
        "mean_test_accuracy": bag_summary["mean_test_accuracy"],
        "std_test_accuracy": bag_summary["std_test_accuracy"]
    })
    print("Finished depth:", depth)

depth_results = pd.DataFrame(depth_rows)
display(depth_results)

depth_pivot = depth_results.pivot(index="depth", columns="model", values="mean_test_accuracy")
depth_pivot["bagging_accuracy_gain"] = (
    depth_pivot["Bagging (100 trees)"] - depth_pivot["Single tree"]
)
display(depth_pivot)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for metric, ax in zip(["bias_squared", "variance", "mean_test_accuracy"], axes):
    for model_name in depth_results["model"].unique():
        subset = depth_results[depth_results["model"] == model_name]
        ax.plot(subset["depth"].astype(str), subset[metric], marker="o", label=model_name)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Depth")
    ax.grid(True, alpha=0.3)
axes[-1].legend()
plt.tight_layout()
plt.show()

final_summary = model_test_results[["model", "clean_accuracy", "masked_accuracy"]].copy()
bv_lookup = bias_variance_results.set_index("model")
final_summary["bias_squared"] = final_summary["model"].map(bv_lookup["bias_squared"])
final_summary["variance"] = final_summary["model"].map(bv_lookup["variance"])
final_summary["mean_test_accuracy_30_refits"] = final_summary["model"].map(bv_lookup["mean_test_accuracy"])
final_summary["std_test_accuracy_30_refits"] = final_summary["model"].map(bv_lookup["std_test_accuracy"])
display(final_summary)

# Save the numeric results so they can be inserted into the report.
tree_results.to_csv("q1_tree_grid.csv", index=False)
bag_results.to_csv("q1_bagging_results.csv", index=False)
disagreement_results.to_csv("q1_disagreement_results.csv", index=False)
q1d_results.to_csv("q1_clean_masked_results.csv", index=False)
boost_results.to_csv("q2_boosting_grid.csv", index=False)
comparison_results.to_csv("q2_bagging_boosting_comparison.csv", index=False)
model_test_results.to_csv("q2_model_test_results.csv", index=False)
bias_variance_results.drop(columns=["accuracies"], errors="ignore").to_csv("q3_bias_variance.csv", index=False)
depth_results.to_csv("q3_depth_results.csv", index=False)
final_summary.to_csv("final_summary.csv", index=False)

print("Saved CSV result tables in the current notebook folder.")
