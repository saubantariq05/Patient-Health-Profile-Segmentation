from pathlib import Path
import joblib
import pandas as pd

# Run this file from the project root:
# python validate_results.py

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "HighBP", "HighChol", "BMI", "Stroke", "HeartDiseaseorAttack",
    "GenHlth", "DiffWalk", "Smoker", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "MentHlth", "PhysHlth", "Age",
]

BINARY_FEATURES = [
    "HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk",
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
]

DIABETES_LABELS = {
    0: "No diabetes reported",
    1: "Prediabetes reported",
    2: "Diabetes reported",
}

def main():
    for path in [
        DATA_PATH,
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "kmeans_model.pkl",
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                "Check that this script is in your project root and that "
                "the CSV and models/scaler.pkl + models/kmeans_model.pkl exist."
            )

    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model = joblib.load(MODEL_DIR / "kmeans_model.pkl")

    required = FEATURES + ["Diabetes_012"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if df[FEATURES].isna().any().any():
        raise ValueError("Missing values exist in the clustering features.")

    # IMPORTANT: Diabetes_012 is not passed into the scaler or K-Means.
    scaled = scaler.transform(df[FEATURES])
    df["Cluster"] = model.predict(scaled)

    print("\n" + "=" * 72)
    print("1. DATASET AND MODEL CHECK")
    print("=" * 72)
    print(f"Dataset shape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Model cluster count: {model.n_clusters}")
    print(f"Features passed to model ({len(FEATURES)}): {FEATURES}")
    print("Diabetes_012 included in model features? NO")

    print("\n" + "=" * 72)
    print("2. CLUSTER SIZES")
    print("=" * 72)
    sizes = df["Cluster"].value_counts().sort_index()
    shares = df["Cluster"].value_counts(normalize=True).sort_index() * 100
    size_table = pd.DataFrame({
        "Respondents": sizes,
        "Dataset share (%)": shares.round(2),
    })
    print(size_table.to_string())

    print("\n" + "=" * 72)
    print("3. DIABETES-STATUS DISTRIBUTION WITHIN EACH CLUSTER")
    print("=" * 72)
    for cluster_id in sorted(df["Cluster"].unique()):
        group = df[df["Cluster"] == cluster_id]
        counts = group["Diabetes_012"].value_counts().reindex([0, 1, 2], fill_value=0)
        percentages = (
            group["Diabetes_012"].value_counts(normalize=True)
            .reindex([0, 1, 2], fill_value=0) * 100
        )

        print(f"\nCluster {cluster_id} — n={len(group):,}")
        for code in [0, 1, 2]:
            print(
                f"  {DIABETES_LABELS[code]}: "
                f"{int(counts[code]):,} ({percentages[code]:.2f}%)"
            )

    print("\n" + "=" * 72)
    print("4. HEALTH AND LIFESTYLE PROFILE")
    print("=" * 72)
    for cluster_id in sorted(df["Cluster"].unique()):
        group = df[df["Cluster"] == cluster_id]
        print(f"\nCluster {cluster_id} — n={len(group):,}")
        for feature in BINARY_FEATURES:
            print(f"  {feature}: {group[feature].mean() * 100:.2f}% Yes")
        print(f"  BMI mean: {group['BMI'].mean():.2f}")
        print(f"  GenHlth mean: {group['GenHlth'].mean():.2f}")
        print(f"  PhysHlth mean: {group['PhysHlth'].mean():.2f}")
        print(f"  MentHlth mean: {group['MentHlth'].mean():.2f}")

    print("\n" + "=" * 72)
    print("5. OVERALL DIABETES DISTRIBUTION (REFERENCE)")
    print("=" * 72)
    overall_counts = df["Diabetes_012"].value_counts().reindex([0, 1, 2], fill_value=0)
    overall_pct = (
        df["Diabetes_012"].value_counts(normalize=True)
        .reindex([0, 1, 2], fill_value=0) * 100
    )
    for code in [0, 1, 2]:
        print(
            f"{DIABETES_LABELS[code]}: "
            f"{int(overall_counts[code]):,} ({overall_pct[code]:.2f}%)"
        )

    print("\n" + "=" * 72)
    print("INTERPRETATION CHECK")
    print("=" * 72)
    print(
        "This script recreates cluster assignments using the saved scaler and "
        "saved model. Compare its output with the original notebook. If the "
        "cluster IDs or statistics differ, check that the notebook used the "
        "same saved model, same scaler, same dataset version, same feature "
        "order, and same duplicate-row handling."
    )
    print(
        "Cluster IDs are arbitrary labels: if two runs have identical groups "
        "but different numeric IDs, compare the group profiles rather than "
        "assuming Cluster 0 must always mean the same thing."
    )

if __name__ == "__main__":
    main()
