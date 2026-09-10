import pandas as pd
from sklearn.inspection import permutation_importance

from src.models.train import (
    load_model_ready_data,
    split_features_target_groups,
    identify_feature_types,
    patient_grouped_split,
    build_preprocessor,
    build_and_train_model,
)


def run_ablation():
    # Rebuild the exact same data and split train.py used
    df = load_model_ready_data()
    X, y, groups = split_features_target_groups(df)
    X_train, X_test, y_train, y_test = patient_grouped_split(X, y, groups)

    # Drop the two dominant features
    features_to_remove = ["number_inpatient", "discharge_disposition"]
    X_train_ablated = X_train.drop(columns=features_to_remove)
    X_test_ablated = X_test.drop(columns=features_to_remove)

    print(f"[1/3] Ablated features: {features_to_remove}")
    print(f"      X_train_ablated shape: {X_train_ablated.shape}")

    # identify_feature_types must be recomputed on the ABLATED data —
    # the original numerical/categorical lists from train.py still
    # include the two dropped columns and would cause a KeyError
    numerical_features, categorical_features = identify_feature_types(X_train_ablated)
    print(f"[2/3] Numerical: {len(numerical_features)} | Categorical: {len(categorical_features)}")

    # Build and train a fresh pipeline on the reduced feature set —
    # this is a standalone experiment, NOT saved as a new model artifact
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    ablated_pipeline = build_and_train_model(preprocessor, X_train_ablated, y_train)

    # Permutation importance on the ablated model
    perm_result = permutation_importance(
        ablated_pipeline, X_test_ablated, y_test,
        n_repeats=10, random_state=42, scoring="average_precision", n_jobs=-1
    )

    ablated_importance = pd.DataFrame({
        "Feature": X_test_ablated.columns,
        "Importance": perm_result.importances_mean
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("[3/3] Top 15 features WITHOUT number_inpatient/discharge_disposition:")
    print(ablated_importance.head(15).to_string(index=False))

    # Specifically surface where A1Cresult landed
    a1c_rank = ablated_importance.index[ablated_importance["Feature"] == "A1Cresult"].tolist()
    if a1c_rank:
        print(f"\nA1Cresult rank in ablated model: #{a1c_rank[0] + 1} of {len(ablated_importance)}")
    else:
        print("\nA1Cresult not found in feature list — check column name")

    return ablated_importance


if __name__ == "__main__":
    run_ablation()