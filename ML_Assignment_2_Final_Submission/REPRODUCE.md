# Exact reproduction commands

Run from the repository root using Python 3.12.

## Option A: pip
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/Q1_Decision_Tree_Bagging.py 2>&1 | tee results/q1.log
python src/Q2_SAMME_Boosting.py 2>&1 | tee results/q2.log
python src/Q3_Bias_Variance.py 2>&1 | tee results/q3.log
```

PowerShell log alternative:
```powershell
python src/Q1_Decision_Tree_Bagging.py *> results/q1.log
python src/Q2_SAMME_Boosting.py *> results/q2.log
python src/Q3_Bias_Variance.py *> results/q3.log
```

## Option B: conda
```bash
conda env create -f environment.yml
conda activate ml-assignment-2
python src/Q1_Decision_Tree_Bagging.py
python src/Q2_SAMME_Boosting.py
python src/Q3_Bias_Variance.py
```

The first run may download Olivetti Faces using scikit-learn. The scripts create the saved split file in their current working directory. Run from the repository root if you want the split/results there. Q2 and Q3 repeat prerequisite computations, so they take longer.

After execution, move/copy `olivetti_split_seed_26006.npz` to `data/` and archive all generated CSVs and logs under `results/`. Do not commit generated numeric claims until you have verified the runs.
