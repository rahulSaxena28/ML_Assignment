"""Q1 — Decision-tree baseline and bagging from scratch
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
