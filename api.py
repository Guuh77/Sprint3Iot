from pathlib import Path
import json
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS

from ai_core import carregar_metricas, carregar_modelo, prever_evasao, validar_entrada


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STUDENTS_PATH = DATA_DIR / "alunos_cadastrados.json"


app = Flask(__name__)
CORS(app)
modelo = carregar_modelo()


def carregar_alunos():
    if not STUDENTS_PATH.exists():
        return []

    with open(STUDENTS_PATH, "r", encoding="utf-8") as students_file:
        return json.load(students_file)


def salvar_alunos(alunos):
    DATA_DIR.mkdir(exist_ok=True)
    with open(STUDENTS_PATH, "w", encoding="utf-8") as students_file:
        json.dump(alunos, students_file, indent=2, ensure_ascii=False)


def montar_aluno(payload):
    nome = str(payload.get("nome", "")).strip()
    if not nome:
        raise ValueError("nome e obrigatorio.")

    return {
        "id": str(uuid.uuid4()),
        "nome": nome,
        **validar_entrada(payload),
    }


def analisar_aluno(aluno):
    resultado = prever_evasao(modelo, aluno)
    return {
        "id": aluno.get("id"),
        "nome": aluno.get("nome", "Aluno"),
        **resultado,
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "OracleLearn IA Evasao"})


@app.get("/model-info")
def model_info():
    metrics = carregar_metricas()
    return jsonify(
        {
            "modelo": "RandomForestClassifier",
            "problema": "classificacao_binaria_evasao",
            "features": [
                "horas_estudadas",
                "exercicios_concluidos",
                "media_notas",
                "dias_inativos",
            ],
            "metricas": metrics,
        }
    )


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        return jsonify(prever_evasao(modelo, payload))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/predict-batch")
def predict_batch():
    try:
        payload = request.get_json(force=True)
        alunos = payload.get("alunos", [])
        if not isinstance(alunos, list) or not alunos:
            return jsonify({"error": "Envie uma lista nao vazia em `alunos`."}), 400

        resultados = [
            {
                "nome": aluno.get("nome", f"Aluno {index + 1}"),
                **prever_evasao(modelo, aluno),
            }
            for index, aluno in enumerate(alunos)
        ]
        return jsonify({"resultados": resultados})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.get("/students")
def list_students():
    return jsonify({"alunos": carregar_alunos()})


@app.post("/students")
def create_student():
    try:
        payload = request.get_json(force=True)
        aluno = montar_aluno(payload)
        alunos = carregar_alunos()
        alunos.append(aluno)
        salvar_alunos(alunos)
        return jsonify(aluno), 201
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.delete("/students/<student_id>")
def delete_student(student_id):
    alunos = carregar_alunos()
    alunos_filtrados = [aluno for aluno in alunos if aluno.get("id") != student_id]

    if len(alunos_filtrados) == len(alunos):
        return jsonify({"error": "Aluno nao encontrado."}), 404

    salvar_alunos(alunos_filtrados)
    return jsonify({"deleted": True})


@app.delete("/students")
def clear_students():
    salvar_alunos([])
    return jsonify({"deleted": True})


@app.post("/students/predict")
def predict_students():
    alunos = carregar_alunos()
    if not alunos:
        return jsonify({"error": "Nenhum aluno cadastrado para analisar."}), 400

    resultados = [analisar_aluno(aluno) for aluno in alunos]
    return jsonify({"resultados": resultados})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
