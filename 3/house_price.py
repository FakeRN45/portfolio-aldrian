import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = np.log1p(train["SalePrice"])
train.drop("SalePrice", axis=1, inplace=True)

all_data = pd.concat([train, test], axis=0, ignore_index=True)

for col in ["GarageYrBlt","GarageArea","GarageCars","BsmtFinSF1","BsmtFinSF2",
            "BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","MasVnrArea"]:
    all_data[col] = all_data[col].fillna(0)

for col in ["GarageType","GarageFinish","GarageQual","GarageCond",
            "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
            "MasVnrType","FireplaceQu","PoolQC","Fence","MiscFeature","Alley"]:
    all_data[col] = all_data[col].fillna("None")

all_data = all_data.fillna(all_data.mode().iloc[0])

all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data["Age"] = all_data["YrSold"] - all_data["YearBuilt"]
all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed = skewed_feats[abs(skewed_feats) > 0.75].index
all_data[skewed] = np.log1p(all_data[skewed])

all_data = pd.get_dummies(all_data)

X = all_data.iloc[:len(y), :]
X_test = all_data.iloc[len(y):, :]

lasso = make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-4, -0.5, 30), cv=5))
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5))
elastic = make_pipeline(StandardScaler(), ElasticNetCV(alphas=np.logspace(-4, -0.5, 30), cv=5, l1_ratio=[.1,.5,.9]))

xgb_model = xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.05,
    max_depth=3, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=0.1,
    reg_lambda=0.6, random_state=42
)

lgb_model = lgb.LGBMRegressor(
    objective="regression", num_leaves=4,
    learning_rate=0.01, n_estimators=5000,
    max_bin=200, bagging_fraction=0.75,
    bagging_freq=5, bagging_seed=7,
    feature_fraction=0.2, feature_fraction_seed=7,
    verbose=-1
)

cat_model = CatBoostRegressor(
    iterations=2000, learning_rate=0.05,
    depth=6, l2_leaf_reg=10,
    eval_metric="RMSE", verbose=False,
    random_state=42
)

stack = StackingRegressor(
    estimators=[
        ('ridge', ridge),
        ('lasso', lasso),
        ('elastic', elastic),
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=RidgeCV()
)

def rmsle_cv(model):
    kf = KFold(10, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse

score = rmsle_cv(stack)
print("Stacking RMSE (CV):", score.mean())

stack.fit(X, y)
final_preds = np.expm1(stack.predict(X_test))

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": final_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")