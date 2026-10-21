"""Generate report figures from CSV files produced by the experiment scripts."""
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

def plot_if_exists(filename, x, y, title, output, group=None):
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        print("Skipping missing result:", path)
        return
    df = pd.read_csv(path)
    if x not in df.columns or y not in df.columns:
        print("Skipping; required columns not found:", filename)
        return
    plt.figure(figsize=(6, 4))
    if group and group in df.columns:
        for label, subset in df.groupby(group):
            plt.plot(subset[x], subset[y], marker="o", label=str(label))
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y.replace("_", " "))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, output), dpi=200)
    plt.close()
    print("Saved", output)

plot_if_exists("q1_bagging_results.csv", "B", "oob_accuracy",
               "Bagging: OOB accuracy", "q1_oob_accuracy.png")
plot_if_exists("q1_disagreement_results.csv", "B", "pairwise_disagreement",
               "Bagging: pairwise disagreement", "q1_disagreement.png")
plot_if_exists("q2_bagging_boosting_comparison.csv", "learners", "clean_test_accuracy",
               "Clean test accuracy: bagging vs SAMME", "q2_comparison.png", group="method")
plot_if_exists("q3_depth_results.csv", "depth", "variance",
               "Bias–variance across tree depths", "q3_variance_depth.png", group="model")
