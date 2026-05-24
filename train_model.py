from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
FEATURES = ["horas_estudadas", "exercicios_concluidos", "media_notas", "dias_inativos"]


def gerar_dados_sinteticos(qtd_alunos=5000):
    rng = np.random.default_rng(42)

    horas_estudadas = rng.integers(1, 120, size=qtd_alunos)
    exercicios_concluidos = rng.integers(0, 80, size=qtd_alunos)
    media_notas = rng.uniform(2.0, 10.0, size=qtd_alunos).round(2)
    dias_inativos = rng.integers(0, 90, size=qtd_alunos)

    score_risco = (
        dias_inativos * 0.48
        - horas_estudadas * 0.18
        - exercicios_concluidos * 0.24
        - media_notas * 1.65
        + rng.normal(0, 4.5, size=qtd_alunos)
    )

    limite = np.percentile(score_risco, 72)
    evasao = (score_risco > limite).astype(int)

    return pd.DataFrame(
        {
            "horas_estudadas": horas_estudadas,
            "exercicios_concluidos": exercicios_concluidos,
            "media_notas": media_notas,
            "dias_inativos": dias_inativos,
            "evasao": evasao,
        }
    )


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    print("1. Gerando dataset sintetico de comportamento de alunos...")
    df = gerar_dados_sinteticos()
    dataset_path = DATA_DIR / "alunos_evasao_sintetico.csv"
    df.to_csv(dataset_path, index=False)

    X = df[FEATURES]
    y = df["evasao"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Treinando Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": dict(
            sorted(
                zip(FEATURES, [round(float(value), 4) for value in model.feature_importances_]),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "sample_predictions": [
            {
                "entrada": X_test.iloc[index].to_dict(),
                "real": int(y_test.iloc[index]),
                "previsto": int(y_pred[index]),
                "probabilidade_evasao": round(float(y_proba[index]), 4),
            }
            for index in range(5)
        ],
    }

    model_path = MODEL_DIR / "modelo_evasao.pkl"
    metrics_path = METRICS_DIR / "model_metrics.json"

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    print(f"3. Modelo salvo em: {model_path}")
    print(f"4. Dataset salvo em: {dataset_path}")
    print(f"5. Metricas salvas em: {metrics_path}")
    print(f"Acuracia no conjunto de teste: {metrics['accuracy']:.2%}")
    print("Importancia das variaveis:")
    for feature, importance in metrics["feature_importance"].items():
        print(f"   - {feature}: {importance}")


if __name__ == "__main__":
    main()
