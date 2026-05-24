from pathlib import Path
import json
import pickle

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_evasao.pkl"
METRICS_PATH = BASE_DIR / "metrics" / "model_metrics.json"
FEATURES = ["horas_estudadas", "exercicios_concluidos", "media_notas", "dias_inativos"]


def carregar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo nao encontrado. Execute `python train_model.py` antes de iniciar a API."
        )

    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


def carregar_metricas():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def validar_entrada(payload):
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        raise ValueError(f"Campos obrigatorios ausentes: {', '.join(missing)}")

    values = {
        "horas_estudadas": float(payload["horas_estudadas"]),
        "exercicios_concluidos": float(payload["exercicios_concluidos"]),
        "media_notas": float(payload["media_notas"]),
        "dias_inativos": float(payload["dias_inativos"]),
    }

    if values["horas_estudadas"] < 0:
        raise ValueError("horas_estudadas deve ser maior ou igual a zero.")
    if values["exercicios_concluidos"] < 0:
        raise ValueError("exercicios_concluidos deve ser maior ou igual a zero.")
    if not 0 <= values["media_notas"] <= 10:
        raise ValueError("media_notas deve estar entre 0 e 10.")
    if values["dias_inativos"] < 0:
        raise ValueError("dias_inativos deve ser maior ou igual a zero.")

    return values


def recomendacao_por_risco(probabilidade):
    if probabilidade >= 75:
        return "Acionar tutor, oferecer monitoria e enviar mensagem personalizada ainda hoje."
    if probabilidade >= 45:
        return "Monitorar aluno, sugerir revisao de conteudo e acompanhar acessos nos proximos dias."
    return "Manter acompanhamento regular e incentivar continuidade da trilha."


def classificar_risco(probabilidade):
    if probabilidade >= 75:
        return "alto"
    if probabilidade >= 45:
        return "medio"
    return "baixo"


def prever_evasao(modelo, payload):
    values = validar_entrada(payload)
    frame = pd.DataFrame([values], columns=FEATURES)
    classe = int(modelo.predict(frame)[0])
    probabilidade = round(float(modelo.predict_proba(frame)[0][1]) * 100, 2)

    return {
        "classe": classe,
        "risco": classificar_risco(probabilidade),
        "probabilidade_evasao": probabilidade,
        "recomendacao": recomendacao_por_risco(probabilidade),
        "entrada": values,
    }
