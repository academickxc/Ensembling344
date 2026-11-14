from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from preprocessing import CV, load_train_data

X_tr, y_tr = load_train_data()

# sanity check
model = RandomForestRegressor()
model.fit(X_tr, y_tr)
print(model.score(X_tr, y_tr))

# cross_validator = GridSearchCV(
#     RandomForestRegressor(),
#     param_grid={
#         "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80],
#         "max_depth": [2, 4, 6, 8],
#         "max_features": ["sqrt", "log2"]
#     },
#     cv=CV,  # timeseriessplit
#     scoring="f1_weighted",
#     n_jobs=-1,
# )

# cross_validator.fit(X_tr, y_tr)

# print(
#     f"Best Parameters as determined by cross-validation: {cross_validator.best_params_}"
# )
