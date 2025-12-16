import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "StudentsPerformance.csv")

    print("Membaca file dari:", data_path)


    df = pd.read_csv(data_path)

    output_dir = os.path.join(
        base_dir,
        "students_performance_preprocessing"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Membaca file dari:", data_path)
    df = pd.read_csv(data_path)

    # ==============================
    # DROP DUPLICATE
    # ==============================
    df = df.drop_duplicates()

    # ==============================
    # TARGET ENCODING (KONSISTEN)
    # ==============================
    df["test_prep"] = np.where(
        df["test preparation course"] == "completed", 1, 0
    )
    df = df.drop(columns=["test preparation course"])

    # ==============================
    # ENCODING KATEGORIK
    # ==============================
    df = pd.get_dummies(df, drop_first=True)

    # ==============================
    # SCALING
    # ==============================
    X = df.drop("test_prep", axis=1)
    y = df["test_prep"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final["test_prep"] = y.values

    # ==============================
    # SAVE
    # ==============================
    output_path = os.path.join(
        output_dir,
        "students_clean.csv"
    )
    df_final.to_csv(output_path, index=False)

    print("Dataset preprocessing berhasil disimpan di:", output_path)


if __name__ == "__main__":
    preprocess_data()
