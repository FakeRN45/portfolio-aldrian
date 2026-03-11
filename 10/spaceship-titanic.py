import gc
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def rare_label_collapse(series: pd.Series, min_count: int = 50, other_label: str = "Other"):
    counts = series.value_counts(dropna=False)
    rare = counts[counts < min_count].index
    return series.where(~series.isin(rare), other_label)

def safe_div(numer, denom):
    denom = np.where(denom == 0, 1e-9, denom)
    return numer / denom

def add_spend_proportions(df, spend_cols, total_col="TotalSpend"):
    for c in spend_cols:
        df[f"Prop_{c}"] = safe_div(df[c].fillna(0).astype(float), (df[total_col].fillna(0).astype(float) + 1e-6))

def log1p_cols(df, cols):
    for c in cols:
        df[f"log1p_{c}"] = np.log1p(df[c].fillna(0).astype(float))

def bin_age(df, col="Age"):
    df["Age_bin"] = pd.cut(df[col], bins=[-1, 0, 12, 18, 25, 40, 60, 200], labels=False)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Cabin_num"] = pd.to_numeric(df["Cabin_num"], errors="coerce")
    
    grp = df["PassengerId"].str.split("_", expand=True)
    df["Group"] = grp[0].astype(int)
    df["Group_idx"] = grp[1].astype(int)
    df["Group_size"] = df["Group"].map(df["Group"].value_counts())
    df["IsAlone"] = (df["Group_size"] == 1)
    
    df["Surname"] = df["Name"].fillna("").apply(lambda x: str(x).split()[-1] if str(x).strip() else "Unknown")
    df["Surname_freq"] = df["Surname"].map(df["Surname"].value_counts())
    df["NameLength"] = df["Name"].fillna("").apply(lambda x: len(str(x)))
    df["SurnameLength"] = df["Surname"].apply(lambda x: len(str(x)))
    
    spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["ZeroSpend"] = (df[spend_cols].fillna(0).sum(axis=1) == 0)
    df["AllZeroSpend"] = ((df[spend_cols].fillna(0) == 0).all(axis=1)).astype(int)
    add_spend_proportions(df, spend_cols, total_col="TotalSpend")
    log1p_cols(df, spend_cols + ["TotalSpend"])
    
    df["SpendPerAge"] = safe_div(df["TotalSpend"], (df["Age"].fillna(0) + 1))
    df["SpendPerMember"] = safe_div(df["TotalSpend"], df["Group_size"].replace(0,1))
    
    df["CryoSleep"] = df["CryoSleep"].fillna(False)
    df["VIP"] = df["VIP"].fillna(False)
    df["SleepMismatch"] = (df["CryoSleep"].astype(bool) & (df["TotalSpend"] > 0))
    
    df["Side_bin"] = df["Side"].map({"P": 0, "S": 1})
    
    df["HomePlanet"] = df["HomePlanet"].fillna("Unknown")
    df["Destination"] = df["Destination"].fillna("Unknown")
    df["Deck"] = df["Deck"].fillna("Unknown")
    df["Side"] = df["Side"].fillna("Unknown")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Cabin_num"] = df["Cabin_num"].fillna(df["Cabin_num"].median())
    df["Surname"] = df["Surname"].fillna("Unknown")
    df["Surname_freq"] = df["Surname_freq"].fillna(1)
    
    bin_age(df, "Age")
    
    df["Surname"] = rare_label_collapse(df["Surname"].astype(str), min_count=5)
    
    for col in ["TotalSpend", "Age"]:
        df[f"Group_{col}_mean"] = df.groupby("Group")[col].transform("mean")
        df[f"Group_{col}_median"] = df.groupby("Group")[col].transform("median")
    
    df["GroupCryoRate"] = df.groupby("Group")["CryoSleep"].transform("mean")
    df["GroupHasVIP"] = df.groupby("Group")["VIP"].transform("max")
    
    deck_max = df.groupby("Deck")["Cabin_num"].transform("max").replace(0, np.nan)
    df["Cabin_rel_pos"] = safe_div(df["Cabin_num"].fillna(0), (deck_max.fillna(df["Cabin_num"].max()) + 1e-6))
    
    df["DeckFreq"] = df["Deck"].map(df["Deck"].value_counts())
    
    return df

def label_encode_fit_transform(train_col, test_col):
    le = LabelEncoder()
    all_vals = pd.concat([train_col.astype(str), test_col.astype(str)], axis=0).fillna("")
    le.fit(all_vals)
    return le.transform(train_col.astype(str)), le.transform(test_col.astype(str))

def build_feature_matrix(train, test):
    train_fe = feature_engineering(train.copy())
    test_fe = feature_engineering(test.copy())
    
    for c in ["HomePlanet", "Destination", "Deck"]:
        all_vals = pd.concat([train_fe[c].astype(str), test_fe[c].astype(str)], axis=0)
        rare = all_vals.value_counts()
        rare_idx = rare[rare < 30].index
        train_fe[c] = train_fe[c].where(~train_fe[c].isin(rare_idx), "Other")
        test_fe[c] = test_fe[c].where(~test_fe[c].isin(rare_idx), "Other")
    
    cat_cols = ["HomePlanet","Destination","Deck","Side","Surname"]
    for col in cat_cols:
        train_fe[col], test_fe[col] = label_encode_fit_transform(train_fe[col], test_fe[col])
    
    spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    base_cols = [
        "HomePlanet","Destination","Deck","Side","Side_bin","Cabin_num","Cabin_rel_pos",
        "Age","Age_bin","VIP","CryoSleep","SleepMismatch",
        "Group","Group_idx","Group_size","IsAlone","Surname","Surname_freq",
        "TotalSpend","ZeroSpend","AllZeroSpend","SpendPerAge","SpendPerMember",
        "NameLength","SurnameLength","DeckFreq",
        "Group_TotalSpend_mean","Group_TotalSpend_median","Group_Age_mean","Group_Age_median",
        "GroupCryoRate","GroupHasVIP"
    ] + spend_cols + [f"Prop_{c}" for c in spend_cols] + [f"log1p_{c}" for c in spend_cols + ["TotalSpend"]]
    
    for b in ["VIP","CryoSleep","SleepMismatch","IsAlone","ZeroSpend","AllZeroSpend","GroupHasVIP"]:
        train_fe[b] = train_fe[b].astype(int)
        test_fe[b] = test_fe[b].astype(int)
    
    X = train_fe[base_cols].copy()
    X_test = test_fe[base_cols].copy()
    for df in [X, X_test]:
        for c in df.columns:
            if df[c].isnull().any():
                if df[c].dtype.kind in "biufc":
                    df[c].fillna(df[c].median(), inplace=True)
                else:
                    df[c].fillna(0, inplace=True)
    
    return X, X_test, train_fe, test_fe

def get_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=3000, learning_rate=0.02,
            num_leaves=96, subsample=0.85, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.2, random_state=42
        ),
        "xgb": XGBClassifier(
            n_estimators=3000, learning_rate=0.02, max_depth=7,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=2.0,
            reg_lambda=1.0, eval_metric="logloss", random_state=42,
            tree_method="hist"
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=18, min_samples_split=8,
            min_samples_leaf=4, n_jobs=-1, random_state=42
        ),
        "et": ExtraTreesClassifier(
            n_estimators=500, min_samples_split=4,
            min_samples_leaf=2, n_jobs=-1, random_state=42
        )
    }

def run_stacking(train_path, test_path, sample_path, out_path="submission-3.csv", n_splits=7, seed=42):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    
    X, X_test, train_fe, test_fe = build_feature_matrix(train, test)
    y = train["Transported"].astype(int).values
    groups = train_fe["Group"].values
    
    gkf = GroupKFold(n_splits=n_splits)
    models = get_models()
    model_names = list(models.keys())
    
    oof_meta = np.zeros((len(X), len(model_names)))
    test_meta = np.zeros((len(X_test), len(model_names)))
    
    print(f"Training base models with {n_splits}-fold GroupKFold...")
    for fold, (trn_idx, val_idx) in enumerate(tqdm(gkf.split(X, y, groups=groups), total=n_splits)):
        X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]
        
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_tr, y_tr)
            oof_meta[val_idx, i] = model.predict_proba(X_va)[:, 1]
            test_meta[:, i] += model.predict_proba(X_test)[:, 1] / n_splits
        
        val_avg = oof_meta[val_idx, :].mean(axis=1)
        print(f"Fold {fold+1} base-avg Accuracy: {accuracy_score(y_va, np.rint(val_avg)):.5f}")
        gc.collect()
    
    base_cv_acc = accuracy_score(y, np.rint(oof_meta.mean(axis=1)))
    print(f"\nBase models average OOF Accuracy: {base_cv_acc:.5f}")
    
    meta = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=16,
        subsample=0.9, colsample_bytree=0.9, random_state=seed
    )
    meta.fit(oof_meta, y)
    meta_oof = meta.predict_proba(oof_meta)[:,1]
    meta_acc = accuracy_score(y, np.rint(meta_oof))
    print(f"Meta-model OOF Accuracy: {meta_acc:.5f}")
    
    blend = 0.7 * meta_oof + 0.3 * oof_meta.mean(axis=1)
    blend_acc = accuracy_score(y, np.rint(blend))
    print(f"Blended OOF Accuracy: {blend_acc:.5f}")
    
    meta_test = meta.predict_proba(test_meta)[:,1]
    final_test = 0.7 * meta_test + 0.3 * test_meta.mean(axis=1)
    
    sub = sample_submission.copy()
    sub["Transported"] = np.rint(final_test).astype(bool)
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission saved as {out_path}")
    
    return {
        "base_avg_oof_acc": base_cv_acc,
        "meta_oof_acc": meta_acc,
        "blend_oof_acc": blend_acc
    }

if __name__ == "__main__":
    train_path = "train.csv"
    test_path = "test.csv"
    sample_path = "sample_submission.csv"
    
    scores = run_stacking(train_path, test_path, sample_path, out_path="submission-3.csv", n_splits=7, seed=42)
    print("\n=== Score Summary ===")
    for k, v in scores.items():
        print(f"{k}: {v:.5f}")