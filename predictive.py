import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def generate_synthetic(n_samples=1000, n_features=10, imbalance=0.9, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    # create binary target with given imbalance (proportion of zeros)
    probs = rng.rand(n_samples)
    y = (probs > imbalance).astype(int)
    cols = [f"sensor_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["failure"] = y
    return df


def load_dataframe(file_path, generate_if_missing=False):
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded dataset from: {file_path} (shape={df.shape})")
            return df
        except Exception as e:
            print(f"Error reading CSV '{file_path}': {e}")
            sys.exit(1)
    if generate_if_missing:
        print("File not found — generating synthetic dataset for demo.")
        return generate_synthetic()
    print(f"File not found: {file_path}. Use --generate-synthetic to create a demo dataset.")
    sys.exit(1)


def prepare_and_train(df, args):
    if "failure" not in df.columns:
        print("ERROR: Column 'failure' not found in dataset.")
        print("Columns available:", list(df.columns))
        sys.exit(1)

    # Basic cleaning
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()

    print("Dataset shape after cleaning:", df.shape)

    X = df.drop("failure", axis=1)
    y = df["failure"]

    # Only use numeric columns for modeling
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        print("No numeric features available for training.")
        sys.exit(1)

    # Handle stratify only when there are at least two classes
    stratify_arg = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify_arg
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        class_weight="balanced" if args.class_weight else None,
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    ax = feat_imp.plot(kind="bar", figsize=(10, 5), title="Feature Importance")
    plt.tight_layout()
    if args.save_plots:
        out_path = args.save_plots
        plt.savefig(out_path)
        print(f"Saved feature importance plot to {out_path}")
    else:
        plt.show()

    # Demonstrate a single sample prediction if available
    if len(X_test) > 0:
        sample = X_test.iloc[0:1]
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        print("Sample prediction:", int(prediction[0]))


def parse_args():
    p = argparse.ArgumentParser(description="Predictive maintenance demo script")
    p.add_argument("--file", "-f", help="Path to CSV file (must contain 'failure' column)")
    p.add_argument("--generate-synthetic", action="store_true", help="Generate a synthetic dataset if file missing")
    p.add_argument("--n-estimators", dest="n_estimators", type=int, default=100)
    p.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    p.add_argument("--random-state", dest="random_state", type=int, default=42)
    p.add_argument("--class-weight", dest="class_weight", action="store_true", help="Use balanced class weight")
    p.add_argument("--save-plots", dest="save_plots", help="Path to save plots (PNG)")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_dataframe(args.file, generate_if_missing=args.generate_synthetic)
    prepare_and_train(df, args)


if __name__ == "__main__":
    main()
