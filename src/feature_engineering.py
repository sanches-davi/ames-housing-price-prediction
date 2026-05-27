from sklearn.model_selection import cross_validate

def create_features(data):

    data = data.copy()
    data["TotalSF"] = (data["Total Bsmt SF"].fillna(0) + data["1st Flr SF"].fillna(0) + data["2nd Flr SF"].fillna(0))
    data["HouseAge"] = (data["Yr Sold"] - data["Year Built"]).clip(lower=0)
    data["RemodAge"] = (data["Yr Sold"] - data["Year Remod/Add"]).clip(lower=0)
    data["TotalBath"] = (
        data["Full Bath"].fillna(0) +
        0.5 * data["Half Bath"].fillna(0) +
        data["Bsmt Full Bath"].fillna(0) +
        0.5 * data["Bsmt Half Bath"].fillna(0)
    )
    data["HasGarage"] = (data["Garage Area"].fillna(0) > 0).astype(int)
    data["HasBasement"] = (data["Total Bsmt SF"].fillna(0) > 0).astype(int)
    data["HasFireplace"] = (data["Fireplaces"].fillna(0) > 0).astype(int)
    data["HasPool"] = (data["Pool Area"].fillna(0) > 0).astype(int)
    
    return data

def evaluate_model_cv(model, X, y, cv=5):
    
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }
    
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    mae = -cv_results["test_MAE"].mean()
    rmse = -cv_results["test_RMSE"].mean()
    r2 = cv_results["test_R2"].mean()
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}