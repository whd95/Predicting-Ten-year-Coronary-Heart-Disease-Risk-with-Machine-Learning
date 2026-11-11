import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv("framingham.csv")
TARGET = "TenYearCHD"

print(df.shape)
print(df[TARGET].value_counts(normalize=True).round(3))

missing = df.isnull().sum()
print(pd.DataFrame({"missing": missing, "percent": (missing / len(df) * 100).round(2)})
      [missing > 0].sort_values("missing", ascending=False))


class FraminghamImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X.median(numeric_only=True)
        self.cigs_smokers_median_ = X.loc[X.currentSmoker == 1, "cigsPerDay"].median()
        self.glucose_by_diabetes_ = X.groupby("diabetes")["glucose"].median().to_dict()
        self.bpmeds_hyp_mode_ = X.loc[X.prevalentHyp == 1, "BPMeds"].mode()[0]
        self.education_mode_ = X["education"].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X.BPMeds.isna() & (X.prevalentHyp == 0), "BPMeds"] = 0
        X["BPMeds"] = X["BPMeds"].fillna(self.bpmeds_hyp_mode_)
        X.loc[X.cigsPerDay.isna() & (X.currentSmoker == 0), "cigsPerDay"] = 0
        X["cigsPerDay"] = X["cigsPerDay"].fillna(self.cigs_smokers_median_)
        by_group = X["diabetes"].map(self.glucose_by_diabetes_)
        X["glucose"] = X["glucose"].fillna(by_group).fillna(self.medians_["glucose"])
        X["education"] = X["education"].fillna(self.education_mode_)
        for col in ["totChol", "BMI", "heartRate"]:
            X[col] = X[col].fillna(self.medians_[col])
        return X


X = df.drop(columns=TARGET)
y = df[TARGET]

imputer = FraminghamImputer().fit(X)
X_clean = imputer.transform(X)

assert X_clean.shape == X.shape
assert X_clean.isnull().sum().sum() == 0
assert (X_clean.loc[X.currentSmoker == 0, "cigsPerDay"] == 0).all()

print("Missing before:", X.isnull().sum().sum(), "| after:", X_clean.isnull().sum().sum())


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

data = X_clean.copy()
data[TARGET] = y.values

num_cols = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
cat_cols = ["male", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]

plt.figure(figsize=(5, 4))
sns.countplot(data=data, x=TARGET)
plt.title("Target balance")
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(data=data, x=col, hue=TARGET, kde=True, stat="density", common_norm=False, ax=ax)
    ax.set_title(f"{col} (skew={data[col].skew():.2f})")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, col in zip(axes.ravel(), num_cols):
    sns.boxplot(data=data, x=TARGET, y=col, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, col in zip(axes.ravel(), cat_cols):
    sns.barplot(data=data, x=col, y=TARGET, errorbar=None, ax=ax)
    ax.axhline(data[TARGET].mean(), color="red", linestyle="--")
    ax.set_title(f"CHD rate by {col}")
axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.show()

data["age_group"] = pd.cut(data["age"], bins=[30, 40, 50, 60, 70], right=False)
plt.figure(figsize=(7, 4))
sns.barplot(data=data, x="age_group", y=TARGET, errorbar=None)
plt.title("CHD rate by age group")
plt.show()

plt.figure(figsize=(12, 9))
sns.heatmap(data.drop(columns="age_group").corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.show()

print(data.drop(columns="age_group").corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3))

print("\nOutliers (IQR rule):")
for col in num_cols:
    q1, q3 = data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col:12s} {n:4d} ({n / len(data) * 100:.1f}%)")


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

SHOW_PLOTS = True

class DistributionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_cols=None, lower=0.01, upper=0.99, skew_threshold=0.5):
        self.clip_cols = clip_cols
        self.lower = lower
        self.upper = upper
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        self.clip_cols_ = self.clip_cols or ["cigsPerDay", "totChol", "sysBP", "diaBP",
                                             "BMI", "heartRate", "glucose"]
        self.low_ = X[self.clip_cols_].quantile(self.lower)
        self.high_ = X[self.clip_cols_].quantile(self.upper)

        Xw = X.copy()
        Xw[self.clip_cols_] = Xw[self.clip_cols_].clip(self.low_, self.high_, axis=1)

        skew = Xw[self.clip_cols_].skew()
        self.skewed_cols_ = skew[skew.abs() > self.skew_threshold].index.tolist()

        self.pt_ = PowerTransformer(method="yeo-johnson", standardize=False)
        self.pt_.fit(Xw[self.skewed_cols_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.clip_cols_] = X[self.clip_cols_].clip(self.low_, self.high_, axis=1)
        X[self.skewed_cols_] = self.pt_.transform(X[self.skewed_cols_])
        return X


dist = DistributionTransformer().fit(X_clean)
X_dist = dist.transform(X_clean)

print("Columns clipped:", dist.clip_cols_)
print("Columns Yeo-Johnson transformed:", dist.skewed_cols_)

clipped = {c: int(((X_clean[c] < dist.low_[c]) | (X_clean[c] > dist.high_[c])).sum())
           for c in dist.clip_cols_}
print("Values clipped per column:", clipped)

skew_report = pd.DataFrame({
    "before": X_clean[num_cols].skew(),
    "after": X_dist[num_cols].skew(),
}).round(2)
print("\nSkewness before vs after:")
print(skew_report)

assert X_dist.shape == X_clean.shape
assert X_dist.isnull().sum().sum() == 0
assert np.isfinite(X_dist.select_dtypes("number").values).all()

if SHOW_PLOTS:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.histplot(X_dist[col], kde=True, ax=ax)
        ax.set_title(f"{col} after (skew={X_dist[col].skew():.2f})")
    plt.tight_layout()
    plt.show()

pipe_prep = Pipeline([("impute", FraminghamImputer()), ("dist", DistributionTransformer())])
X_ready = pipe_prep.fit_transform(X)
print("\nPipeline output shape:", X_ready.shape, "| NaNs:", int(X_ready.isnull().sum().sum()))


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, groups=("bp", "flags", "interactions", "risk_count")):
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "bp" in self.groups:
            X["pulse_pressure"] = X["sysBP"] - X["diaBP"]
            X["mean_arterial_pressure"] = X["diaBP"] + (X["sysBP"] - X["diaBP"]) / 3

        if "flags" in self.groups or "risk_count" in self.groups:
            flags = pd.DataFrame({
                "high_bp": ((X["sysBP"] >= 140) | (X["diaBP"] >= 90)).astype(int),
                "high_glucose": (X["glucose"] >= 126).astype(int),
                "obese": (X["BMI"] >= 30).astype(int),
                "high_chol": (X["totChol"] >= 240).astype(int),
            }, index=X.index)
            if "flags" in self.groups:
                X = pd.concat([X, flags], axis=1)
            if "risk_count" in self.groups:
                X["risk_count"] = (flags.sum(axis=1) + X["diabetes"] + X["prevalentHyp"]
                                   + X["currentSmoker"] + X["prevalentStroke"])

        if "interactions" in self.groups:
            X["age_x_sysBP"] = X["age"] * X["sysBP"]
            X["age_x_male"] = X["age"] * X["male"]
            X["age_x_glucose"] = X["age"] * X["glucose"]
            X["pack_years"] = X["cigsPerDay"] * X["age"] / 20

        return X


BASE_CLIP = ["cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
NEW_CONTINUOUS = {
    "bp": ["pulse_pressure", "mean_arterial_pressure"],
    "interactions": ["age_x_sysBP", "age_x_glucose", "pack_years"],
}


def build_pipeline(groups, model):
    clip_cols = BASE_CLIP + [c for g in groups for c in NEW_CONTINUOUS.get(g, [])]
    return Pipeline([
        ("impute", FraminghamImputer()),
        ("fe", FeatureEngineer(groups)),
        ("dist", DistributionTransformer(clip_cols=clip_cols)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

models = {
    "LogReg": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
    "HGB": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                  class_weight="balanced", random_state=42),
}

feature_sets = {
    "baseline": (),
    "bp": ("bp",),
    "flags": ("flags",),
    "interactions": ("interactions",),
    "risk_count": ("risk_count",),
    "bp+flags+risk": ("bp", "flags", "risk_count"),
    "all": ("bp", "flags", "interactions", "risk_count"),
}

rows = []
for model_name, make_model in models.items():
    for set_name, groups in feature_sets.items():
        scores = cross_val_score(build_pipeline(groups, make_model()), X, y,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": model_name, "features": set_name,
                     "auc_mean": scores.mean(), "auc_std": scores.std()})

fe_results = pd.DataFrame(rows)
print(fe_results.round(4).to_string(index=False))

baseline_auc = fe_results[fe_results.features == "baseline"].set_index("model")["auc_mean"]
fe_results["gain_vs_baseline"] = fe_results.apply(lambda r: r.auc_mean - baseline_auc[r.model], axis=1)
print("\nGain vs baseline (AUC):")
print(fe_results.pivot(index="features", columns="model", values="gain_vs_baseline").round(4))

FE_GROUPS = ()
X_fe = FeatureEngineer(FE_GROUPS).fit_transform(X_clean)
print("\nSelected groups:", FE_GROUPS, "| shape:", X_fe.shape)


from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols=()):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=list(self.cols))


def compute_vif(df):
    vifs = {}
    for col in df.columns:
        others = df.drop(columns=col)
        r2 = LinearRegression().fit(others, df[col]).score(others, df[col])
        vifs[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(vifs).sort_values(ascending=False)


X_vif = pd.DataFrame(StandardScaler().fit_transform(X_dist), columns=X_dist.columns)
print("VIF (values above 5 indicate strong collinearity):")
print(compute_vif(X_vif).round(2))

pca_full = PCA().fit(StandardScaler().fit_transform(X_dist))
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
print("\nPCA cumulative explained variance:")
print(pd.Series(cum_var, index=range(1, len(cum_var) + 1)).round(3).to_string())


def build_pipeline_dr(model, reducer="passthrough", drop=()):
    return Pipeline([
        ("impute", FraminghamImputer()),
        ("fe", FeatureEngineer(FE_GROUPS)),
        ("dist", DistributionTransformer(clip_cols=BASE_CLIP)),
        ("drop", DropColumns(drop)),
        ("scale", StandardScaler()),
        ("reduce", reducer),
        ("model", model),
    ])


reducers = {
    "none": dict(reducer="passthrough"),
    "drop diaBP": dict(drop=("diaBP",)),
    "drop diaBP+currentSmoker": dict(drop=("diaBP", "currentSmoker")),
    "PCA 90%": dict(reducer=PCA(n_components=0.90)),
    "PCA 95%": dict(reducer=PCA(n_components=0.95)),
    "PCA 99%": dict(reducer=PCA(n_components=0.99)),
    "SelectKBest k=8": dict(reducer=SelectKBest(f_classif, k=8)),
    "SelectKBest k=11": dict(reducer=SelectKBest(f_classif, k=11)),
    "Lasso selection": dict(reducer=SelectFromModel(Lasso(alpha=0.005))),
    "RF importance selection": dict(reducer=SelectFromModel(
        RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced",
                               random_state=42), threshold="median")),
}

rows = []
for model_name, make_model in models.items():
    for name, kwargs in reducers.items():
        scores = cross_val_score(build_pipeline_dr(make_model(), **kwargs), X, y,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": model_name, "method": name,
                     "auc_mean": scores.mean(), "auc_std": scores.std()})

dr_results = pd.DataFrame(rows)
print("\n" + dr_results.round(4).to_string(index=False))

base = dr_results[dr_results.method == "none"].set_index("model")["auc_mean"]
dr_results["gain"] = dr_results.apply(lambda r: r.auc_mean - base[r.model], axis=1)
print("\nGain vs no reduction (AUC):")
print(dr_results.pivot(index="method", columns="model", values="gain").round(4))

X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_dist), columns=X_dist.columns)
lasso_mask = SelectFromModel(Lasso(alpha=0.005)).fit(X_scaled, y).get_support()
print("\nLasso keeps:", X_scaled.columns[lasso_mask].tolist())
print("Lasso drops:", X_scaled.columns[~lasso_mask].tolist())

mean_gain = dr_results.groupby("method")["gain"].mean().sort_values(ascending=False)
print("\nMean gain across both models:")
print(mean_gain.round(4))

DR_CHOICE = mean_gain.index[0] if mean_gain.iloc[0] > 0.002 else "none"
DR_KWARGS = reducers[DR_CHOICE]
print("\nSelected:", DR_CHOICE)


from sklearn.base import clone
from sklearn.model_selection import train_test_split

SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print("Train:", X_train.shape, "| Test:", X_test.shape)
print("\nTarget rate (full / train / test):")
print(round(y.mean(), 4), round(y_train.mean(), 4), round(y_test.mean(), 4))
print("Positives in train / test:", int(y_train.sum()), "/", int(y_test.sum()))
print("Missing values in train / test:", int(X_train.isnull().sum().sum()), "/", int(X_test.isnull().sum().sum()))


def build_preprocessor(reducer="passthrough", drop=()):
    return Pipeline([
        ("impute", FraminghamImputer()),
        ("fe", FeatureEngineer(FE_GROUPS)),
        ("dist", DistributionTransformer(clip_cols=BASE_CLIP)),
        ("drop", DropColumns(drop)),
        ("scale", StandardScaler()),
        ("reduce", clone(reducer) if not isinstance(reducer, str) else reducer),
    ])


preprocessor = build_preprocessor(**DR_KWARGS)
print("\nPreprocessing choice:", DR_CHOICE)

X_train_prep = preprocessor.fit_transform(X_train, y_train)
X_test_prep = preprocessor.transform(X_test)

print("Train prepared:", X_train_prep.shape, "| Test prepared:", X_test_prep.shape)
assert not np.isnan(X_train_prep).any() and not np.isnan(X_test_prep).any()

X_tmp = X_train
for step_name in ["impute", "fe", "dist", "drop"]:
    X_tmp = preprocessor.named_steps[step_name].transform(X_tmp)
names_before_reduce = X_tmp.columns

reduce_step = preprocessor.named_steps["reduce"]
if hasattr(reduce_step, "get_support"):
    kept = names_before_reduce[reduce_step.get_support()].tolist()
    dropped = [c for c in names_before_reduce if c not in kept]
    print("\nFeatures kept (selected on train only):", kept)
    print("Features dropped:", dropped)

imp_train = preprocessor.named_steps["impute"]
imp_full = FraminghamImputer().fit(X)
print("\nLeakage check (statistics learned from train only vs full data):")
print("  glucose median by diabetes (train):", imp_train.glucose_by_diabetes_)
print("  glucose median by diabetes (full) :", imp_full.glucose_by_diabetes_)
print("  scaler mean of first 3 columns (train):",
      preprocessor.named_steps["scale"].mean_[:3].round(3))

cv_train = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
folds = list(cv_train.split(X_train, y_train))
print("\nCV folds:", len(folds), "| validation fold size:", len(folds[0][1]),
      "| positives per fold:", int(y_train.iloc[folds[0][1]].sum()))

dummy_acc = max(y_test.mean(), 1 - y_test.mean())
print("\nBaseline accuracy (always predict majority class):", round(dummy_acc, 4))

import joblib
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_validate,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline

N_ITER = 20
THRESHOLD_MODE = "youden"   # "youden" | "f1" | "recall70"


def get_scores(model, X_):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_)[:, 1]
    return model.decision_function(X_)


pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

candidates = {
    "LogReg": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                                                   class_weight="balanced_subsample", random_state=SEED),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                                               class_weight="balanced_subsample", random_state=SEED),
    "HGB": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                  class_weight="balanced", random_state=SEED),
    "SVM": lambda: SVC(class_weight="balanced", random_state=SEED),
}

param_spaces = {
    "LogReg": {"model__C": np.logspace(-3, 1, 20)},
    "RandomForest": {"model__n_estimators": [200, 400], "model__max_depth": [3, 4, 5, 6, 8],
                     "model__min_samples_leaf": [5, 10, 20, 40], "model__max_features": ["sqrt", 0.5]},
    "ExtraTrees": {"model__n_estimators": [200, 400], "model__max_depth": [3, 4, 5, 6, 8],
                   "model__min_samples_leaf": [5, 10, 20, 40], "model__max_features": ["sqrt", 0.5]},
    "HGB": {"model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__max_depth": [2, 3, 4],
            "model__max_iter": [100, 200, 300], "model__l2_regularization": [0, 1, 5, 10],
            "model__min_samples_leaf": [20, 40, 80]},
    "SVM": {"model__C": [0.05, 0.1, 0.5, 1, 2, 5], "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
}

try:
    from xgboost import XGBClassifier

    def make_xgb():
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                             random_state=SEED, n_jobs=1, verbosity=0)

    cross_val_score(make_pipeline(make_xgb()), np.random.rand(40, 3), np.array([0, 1] * 20), cv=2)
    candidates["XGBoost"] = make_xgb
    param_spaces["XGBoost"] = {
        "model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [1, 5, 10],
        "model__reg_lambda": [1, 5, 10],
    }
except Exception as e:
    print("XGBoost skipped:", type(e).__name__)

print("Models:", list(candidates))


dr_options = {"all features": dict(reducer="passthrough"), DR_CHOICE: DR_KWARGS}
scoring = {"auc": "roc_auc", "pr_auc": "average_precision"}

rows = []
for model_name, make_model in candidates.items():
    for dr_name, dr_kwargs in dr_options.items():
        res = cross_validate(build_pipeline_dr(make_model(), **dr_kwargs), X_train, y_train,
                             cv=cv_train, scoring=scoring, n_jobs=-1)
        rows.append({"model": model_name, "features": dr_name,
                     "auc": res["test_auc"].mean(), "auc_std": res["test_auc"].std(),
                     "pr_auc": res["test_pr_auc"].mean()})

baseline_df = pd.DataFrame(rows)
print("\n=== 7b) Baseline CV results ===")
print(baseline_df.round(4).to_string(index=False))
print("PR-AUC of a random model (= positive rate):", round(y_train.mean(), 4))

pivot = baseline_df.pivot(index="model", columns="features", values="auc")
mean_gain = (pivot[DR_CHOICE] - pivot["all features"]).mean()
print(f"\nMean AUC gain of '{DR_CHOICE}' over all features: {mean_gain:.4f}")
FINAL_DR = DR_KWARGS if mean_gain > 0.002 else dict(reducer="passthrough")
print("Feature selection used from now on:", DR_CHOICE if mean_gain > 0.002 else "all features")


print("\n=== 7c) Tuning all models ===")
cv_search = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
tuned = {}
for name in candidates:
    grid_size = int(np.prod([len(v) for v in param_spaces[name].values()]))
    search = RandomizedSearchCV(
        build_pipeline_dr(candidates[name](), **FINAL_DR), param_spaces[name],
        n_iter=min(N_ITER, grid_size), cv=cv_search, scoring="roc_auc",
        n_jobs=-1, random_state=SEED)
    search.fit(X_train, y_train)
    tuned[name] = search.best_estimator_
    print(f"{name:13s} best CV AUC = {search.best_score_:.4f} | {search.best_params_}")


stack = StackingClassifier(
    estimators=[(name, est) for name, est in tuned.items()],
    final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced"),
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    stack_method="auto", n_jobs=1)

voters = [(n, e) for n, e in tuned.items() if hasattr(e, "predict_proba")]
voting = VotingClassifier(estimators=voters, voting="soft", n_jobs=1)


print("\n=== 7e) Final candidates ===")
final_candidates = dict(tuned)
final_candidates["Stacking"] = stack
final_candidates["SoftVoting"] = voting

cv_final = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED + 1)
final_rows = []
for name, est in final_candidates.items():
    res = cross_validate(est, X_train, y_train, cv=cv_final, scoring=scoring, n_jobs=-1)
    final_rows.append({"model": name, "auc": res["test_auc"].mean(), "auc_std": res["test_auc"].std(),
                       "pr_auc": res["test_pr_auc"].mean()})

final_df = pd.DataFrame(final_rows).sort_values("auc", ascending=False)
print(final_df.round(4).to_string(index=False))

best_name = final_df.iloc[0]["model"]
best_model = final_candidates[best_name]
print("\nSelected model:", best_name)

# 7f) Decision threshold (out-of-fold predictions on train)
oof_method = "predict_proba" if hasattr(best_model, "predict_proba") else "decision_function"
oof = cross_val_predict(best_model, X_train, y_train,
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method=oof_method, n_jobs=-1)
if oof.ndim == 2:
    oof = oof[:, 1]


def pick_threshold(y_true, scores, mode):
    if mode == "youden":
        fpr_, tpr_, thr_ = roc_curve(y_true, scores)
        return float(thr_[np.argmax(tpr_ - fpr_)])
    prec, rec, pthr = precision_recall_curve(y_true, scores)
    if mode == "f1":
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        return float(pthr[np.argmax(f1)])
    if mode == "recall70":
        idx = np.where(rec[:-1] >= 0.70)[0]
        return float(pthr[idx[-1]])
    raise ValueError(mode)


print("\nThreshold options (out-of-fold on train):")
for mode in ["youden", "f1", "recall70"]:
    t = pick_threshold(y_train, oof, mode)
    p = (oof >= t).astype(int)
    tp = ((p == 1) & (y_train.values == 1)).sum()
    print(f"  {mode:9s} threshold={t:.3f} | recall={tp / y_train.sum():.3f} | "
          f"precision={tp / max(p.sum(), 1):.3f} | flagged={p.mean():.1%}")

threshold = pick_threshold(y_train, oof, THRESHOLD_MODE)
print("Used:", THRESHOLD_MODE, "->", round(threshold, 3))


print("\n=== 7g) FINAL TEST RESULTS ===")
best_model.fit(X_train, y_train)
scores_test = get_scores(best_model, X_test)
pred = (scores_test >= threshold).astype(int)

auc = roc_auc_score(y_test, scores_test)
rng = np.random.default_rng(SEED)
y_arr = y_test.values
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(y_arr), len(y_arr))
    if y_arr[idx].min() != y_arr[idx].max():
        boot.append(roc_auc_score(y_arr[idx], scores_test[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f"Model   : {best_name}")
print(f"ROC-AUC : {auc:.4f}  (95% CI: {lo:.3f} - {hi:.3f})")
print(f"PR-AUC  : {average_precision_score(y_test, scores_test):.4f}  (random = {y_test.mean():.3f})")
print(f"\nClassification report at threshold {threshold:.3f}:")
print(classification_report(y_test, pred, target_names=["No CHD", "CHD"]))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fpr_t, tpr_t, _ = roc_curve(y_test, scores_test)
axes[0].plot(fpr_t, tpr_t, label=f"AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve (test)")
axes[0].legend()
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[1], colorbar=False)
axes[1].set_title("Confusion matrix (test)")
plt.tight_layout()
plt.show()

perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                              n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()
print("\nPermutation importance (AUC drop on test):")
print(imp.sort_values(ascending=False).round(4))
imp.plot(kind="barh", figsize=(8, 6), title="Permutation importance (test AUC drop)")
plt.tight_layout()
plt.show()

best_model.fit(X, y)
joblib.dump({"model": best_model, "name": best_name, "threshold": threshold,
             "features": X.columns.tolist()}, "framingham_final_model.joblib")
print("\nSaved: framingham_final_model.joblib")

from sklearn.metrics import brier_score_loss


lr_balanced = clone(tuned["LogReg"])
lr_plain = clone(tuned["LogReg"]).set_params(model__class_weight=None)

for label, est in [("balanced weights", lr_balanced), ("no class weights", lr_plain)]:
    res = cross_validate(est, X_train, y_train, cv=cv_final, scoring=scoring, n_jobs=-1)
    print(f"LogReg ({label}): AUC={res['test_auc'].mean():.4f} +/- {res['test_auc'].std():.4f} "
          f"| PR-AUC={res['test_pr_auc'].mean():.4f}")

final_lr = lr_plain


oof = cross_val_predict(final_lr, X_train, y_train,
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method="predict_proba", n_jobs=-1)[:, 1]

print("\nThreshold options (out-of-fold on train):")
for mode in ["youden", "f1", "recall70"]:
    t = pick_threshold(y_train, oof, mode)
    p = (oof >= t).astype(int)
    tp = ((p == 1) & (y_train.values == 1)).sum()
    print(f"  {mode:9s} threshold={t:.3f} | recall={tp / y_train.sum():.3f} | "
          f"precision={tp / max(p.sum(), 1):.3f} | flagged={p.mean():.1%}")

threshold = pick_threshold(y_train, oof, THRESHOLD_MODE)
print("Used:", THRESHOLD_MODE, "->", round(threshold, 3))


final_lr.fit(X_train, y_train)
proba_test = final_lr.predict_proba(X_test)[:, 1]
pred = (proba_test >= threshold).astype(int)

auc = roc_auc_score(y_test, proba_test)
rng = np.random.default_rng(SEED)
y_arr = y_test.values
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(y_arr), len(y_arr))
    if y_arr[idx].min() != y_arr[idx].max():
        boot.append(roc_auc_score(y_arr[idx], proba_test[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])

print("\n=== FINAL TEST RESULTS: Logistic Regression ===")
print(f"ROC-AUC : {auc:.4f}  (95% CI: {lo:.3f} - {hi:.3f})")
print(f"PR-AUC  : {average_precision_score(y_test, proba_test):.4f}  (random = {y_test.mean():.3f})")
print(f"Brier   : {brier_score_loss(y_test, proba_test):.4f}  "
      f"(always predicting the base rate = {brier_score_loss(y_test, np.full(len(y_test), y_train.mean())):.4f})")
print(f"\nClassification report at threshold {threshold:.3f}:")
print(classification_report(y_test, pred, target_names=["No CHD", "CHD"]))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))


calib = pd.DataFrame({"p": proba_test, "y": y_test.values})
calib["risk_group"] = pd.qcut(calib["p"], 5, labels=["lowest", "low", "middle", "high", "highest"],
                              duplicates="drop")
print("\nCalibration by risk group (test):")
print(calib.groupby("risk_group", observed=True).agg(
    n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).round(3))


X_tmp = X_train
for step_name in ["impute", "fe", "dist", "drop"]:
    X_tmp = final_lr.named_steps[step_name].transform(X_tmp)
feature_names = list(X_tmp.columns)
reduce_step = final_lr.named_steps["reduce"]
if hasattr(reduce_step, "get_support"):
    feature_names = [c for c, keep in zip(feature_names, reduce_step.get_support()) if keep]

coefs = final_lr.named_steps["model"].coef_[0]
coef_table = pd.DataFrame({"feature": feature_names, "coef": coefs,
                           "odds_ratio_per_SD": np.exp(coefs)})
coef_table = coef_table.reindex(coef_table["coef"].abs().sort_values(ascending=False).index)
print("\nLogistic Regression coefficients:")
print(coef_table.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
fpr_t, tpr_t, _ = roc_curve(y_test, proba_test)
axes[0].plot(fpr_t, tpr_t, label=f"AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve (test)")
axes[0].legend()
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[1], colorbar=False)
axes[1].set_title("Confusion matrix (test)")
cg = calib.groupby("risk_group", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
axes[2].plot(cg["p"], cg["y"], marker="o", label="Logistic Regression")
axes[2].plot([0, cg.values.max()], [0, cg.values.max()], "--", color="gray", label="Perfect")
axes[2].set_xlabel("Mean predicted risk")
axes[2].set_ylabel("Observed rate")
axes[2].set_title("Calibration (test)")
axes[2].legend()
plt.tight_layout()
plt.show()


final_lr_all = clone(final_lr).fit(X, y)
joblib.dump({"model": final_lr_all, "threshold": threshold, "features": X.columns.tolist()},
            "framingham_logreg_final.joblib")
print("\nSaved: framingham_logreg_final.joblib")


def predict_risk(new_data, path="framingham_logreg_final.joblib"):
    bundle = joblib.load(path)
    new_data = new_data[bundle["features"]]
    prob = bundle["model"].predict_proba(new_data)[:, 1]
    return pd.DataFrame({"risk_probability": prob.round(3),
                         "high_risk": prob >= bundle["threshold"]}, index=new_data.index)


example = pd.DataFrame([{
    "male": 1, "age": 58, "education": 2, "currentSmoker": 1, "cigsPerDay": 20, "BPMeds": 0,
    "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0, "totChol": 240, "sysBP": 150,
    "diaBP": 92, "BMI": 28, "heartRate": 78, "glucose": 95,
}, {
    "male": 0, "age": 38, "education": 3, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 0,
    "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0, "totChol": 190, "sysBP": 115,
    "diaBP": 75, "BMI": 23, "heartRate": 70, "glucose": 85,
}])
print("\nExample predictions:")
print(predict_risk(example))
