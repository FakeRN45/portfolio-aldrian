import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("Loading Sentence-BERT model...")
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print("Fitting TF-IDF vectorizer...")
corpus = []
train_df = pd.read_csv("train.csv")
for idx, row in train_df.iterrows():
    article_id = row['id']
    folder = f"train/article_{int(article_id):04d}"
    f1 = open(f"{folder}/file_1.txt", encoding="utf-8").read()
    f2 = open(f"{folder}/file_2.txt", encoding="utf-8").read()
    corpus.extend([f1, f2])
tfidf = TfidfVectorizer(max_features=10000)
tfidf.fit(corpus)

def jaccard_similarity(s1, s2):
    a, b = set(s1.split()), set(s2.split())
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0

def extract_features_pair(text1, text2):
    emb1 = embedder.encode(text1, convert_to_numpy=True)
    emb2 = embedder.encode(text2, convert_to_numpy=True)
    
    length1, length2 = len(text1.split()), len(text2.split())
    uniq1, uniq2 = len(set(text1.split())), len(set(text2.split()))
    flesch1, flesch2 = textstat.flesch_reading_ease(text1), textstat.flesch_reading_ease(text2)
    fog1, fog2 = textstat.gunning_fog(text1), textstat.gunning_fog(text2)

    feat1 = np.concatenate([emb1, [length1, uniq1, flesch1, fog1]])
    feat2 = np.concatenate([emb2, [length2, uniq2, flesch2, fog2]])

    diff = feat1 - feat2
    diff_rev = feat2 - feat1
    prod = feat1 * feat2
    abs_diff = np.abs(diff)

    tfidf1 = tfidf.transform([text1])
    tfidf2 = tfidf.transform([text2])
    cosine_sim = cosine_similarity(tfidf1, tfidf2)[0][0]

    jaccard_sim = jaccard_similarity(text1, text2)

    return np.concatenate([diff, diff_rev, prod, abs_diff, [cosine_sim, jaccard_sim]])

X, y = [], []

for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting features"):
    article_id = row['id']
    real_id = row['real_text_id']
    folder = f"train/article_{int(article_id):04d}"
    f1 = open(f"{folder}/file_1.txt", encoding="utf-8").read()
    f2 = open(f"{folder}/file_2.txt", encoding="utf-8").read()

    feats = extract_features_pair(f1, f2)
    X.append(feats)
    y.append(1 if real_id == 1 else 0)

X, y = np.array(X), np.array(y)

print("\nRunning Stratified K-Fold Cross-Validation...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_losses, acc_scores = [], []
feature_importance = np.zeros(X.shape[1])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    xgb = XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=10,
                        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                        tree_method="hist", random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    lgbm = LGBMClassifier(n_estimators=700, learning_rate=0.05, max_depth=10, random_state=42)

    xgb.fit(X_train, y_train)
    logreg.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    proba = (
        xgb.predict_proba(X_val) +
        logreg.predict_proba(X_val_scaled) +
        rf.predict_proba(X_val) +
        lgbm.predict_proba(X_val)
    ) / 4

    preds = (proba[:, 1] > 0.5).astype(int)

    acc = accuracy_score(y_val, preds)
    loss = log_loss(y_val, proba[:, 1])

    print(f"Fold {fold} - Accuracy: {acc:.4f} | LogLoss: {loss:.4f}")
    acc_scores.append(acc)
    log_losses.append(loss)

    feature_importance += lgbm.feature_importances_

print("\n========== Overall CV Result ==========")
print(f"Mean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"Mean LogLoss : {np.mean(log_losses):.4f} ± {np.std(log_losses):.4f}")

print("\nPlotting Feature...")
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=[f"F{i}" for i in range(X.shape[1])])
plt.title("Feature Importance (LightGBM)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature plot saved as 'feature_importance.png'")
print("\nTraining final model on full training data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

xgb = XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=10,
                    subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                    tree_method="hist", random_state=42)
logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
lgbm = LGBMClassifier(n_estimators=700, learning_rate=0.05, max_depth=10, random_state=42)

xgb.fit(X, y)
logreg.fit(X_scaled, y)
rf.fit(X, y)
lgbm.fit(X, y)

print("Predicting test set...")
submission = []
test_dir = "test"
folders = sorted([f for f in os.listdir(test_dir) if f.startswith("article_")])

for folder in tqdm(folders, desc="Predicting"):
    article_id = int(folder.split("_")[1])
    
    f1 = open(f"{test_dir}/{folder}/file_1.txt", encoding="utf-8").read()
    f2 = open(f"{test_dir}/{folder}/file_2.txt", encoding="utf-8").read()
    
    feat = extract_features_pair(f1, f2).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    proba = (
        xgb.predict_proba(feat) +
        logreg.predict_proba(feat_scaled) +
        rf.predict_proba(feat) +
        lgbm.predict_proba(feat)
    ) / 4

    pred1 = proba[0][1]

    feat_rev = extract_features_pair(f2, f1).reshape(1, -1)
    feat_rev_scaled = scaler.transform(feat_rev)

    proba_rev = (
        xgb.predict_proba(feat_rev) +
        logreg.predict_proba(feat_rev_scaled) +
        rf.predict_proba(feat_rev) +
        lgbm.predict_proba(feat_rev)
    ) / 4

    pred2 = proba_rev[0][1]

    real_text_id = 1 if pred1 > pred2 else 2
    submission.append((article_id, real_text_id))

sub_df = pd.DataFrame(submission, columns=["id", "real_text_id"])
sub_df = sub_df.sort_values("id")
sub_df.to_csv("submission.csv", index=False)
print("File 'submission.csv' saved!")