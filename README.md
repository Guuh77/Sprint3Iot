# Sprint 4 IoT - OracleLearn IA de Predicao de Evasao

Esta entrega implementa um modelo de Inteligencia Artificial para prever risco de evasao de alunos na plataforma OracleLearn. A integracao com APEX foi substituida, conforme escopo da demonstracao, por uma API Python consumida por um painel Streamlit demonstrativo.

## Objetivo

Identificar alunos com maior probabilidade de abandono com base em indicadores de uso:

- `horas_estudadas`
- `exercicios_concluidos`
- `media_notas`
- `dias_inativos`

O sistema permite cadastrar alunos, salvar seus indicadores, processar a turma em lote e retornar classe prevista, probabilidade de evasao, nivel de risco e recomendacao de intervencao.

## Tecnologias

- Python
- Scikit-learn
- Random Forest Classifier
- Flask + Flask-CORS
- Streamlit

## Estrutura

- `train_model.py`: gera dataset sintetico, treina o modelo e salva metricas.
- `ai_core.py`: centraliza carregamento do modelo, validacao e inferencia.
- `api.py`: API REST consumida pelo painel demonstrativo.
- `app.py`: painel Streamlit demonstrativo.
- `data/`: dataset sintetico gerado.
- `data/alunos_cadastrados.json`: alunos salvos pelo painel/API.
- `models/`: modelo treinado em `.pkl`.
- `metrics/`: metricas, matriz de confusao e importancia das variaveis.

## Como executar

Crie o ambiente e instale dependencias:

```bash
cd Sprint4Iot
pip install -r requirements.txt
```

No Windows/PowerShell, se quiser isolar o ambiente:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Treine o modelo:

```bash
python train_model.py
```

Inicie a API de IA:

```bash
python api.py
```

Teste a API:

```bash
curl http://localhost:8000/health
```

Opcionalmente, rode o painel Streamlit:

```bash
streamlit run app.py
```

No painel, cadastre alunos, veja a turma salva e clique em `Processar turma cadastrada`.

## Endpoints

### GET `/health`

Verifica disponibilidade da API.

### GET `/model-info`

Retorna tipo de modelo, features e metricas salvas.

### POST `/predict`

Entrada:

```json
{
  "horas_estudadas": 8,
  "exercicios_concluidos": 2,
  "media_notas": 3.9,
  "dias_inativos": 35
}
```

Saida:

```json
{
  "classe": 1,
  "risco": "alto",
  "probabilidade_evasao": 98.4,
  "recomendacao": "Acionar tutor, oferecer monitoria e enviar mensagem personalizada ainda hoje."
}
```

### POST `/predict-batch`

Recebe uma lista de alunos e retorna predicoes em lote para simular um painel administrativo.

### GET `/students`

Lista os alunos cadastrados no sistema.

### POST `/students`

Cadastra um aluno e salva os indicadores em `data/alunos_cadastrados.json`.

Entrada:

```json
{
  "nome": "Joao Silva",
  "horas_estudadas": 15,
  "exercicios_concluidos": 5,
  "media_notas": 5.5,
  "dias_inativos": 18
}
```

### DELETE `/students/{id}`

Remove um aluno cadastrado.

### DELETE `/students`

Limpa toda a turma cadastrada.

### POST `/students/predict`

Processa todos os alunos cadastrados, envia os indicadores para o modelo de IA e retorna as recomendacoes por aluno.

## Roteiro para o video

1. Explicar que a IA substitui a analise manual de risco de abandono.
2. Mostrar `train_model.py` treinando o Random Forest e gerando metricas.
3. Rodar `python api.py` e abrir `/health` ou `/model-info`.
4. Abrir o painel Streamlit no navegador.
5. Cadastrar um aluno com bom desempenho e outro aluno com alto numero de dias inativos.
6. Mostrar a tabela de turma cadastrada sendo atualizada.
7. Clicar em `Processar turma cadastrada` e mostrar riscos diferentes no lote.
8. Mostrar a recomendacao gerada para apoiar a decisao do tutor.

## Observacao para entrega

Ao gerar o `.zip`, nao inclua `.venv/`, `__pycache__/` nem arquivos temporarios. O projeto deve levar o codigo-fonte, README, dataset/modelo/metricas gerados e o arquivo com links do video e GitHub.

## Observacao sobre APEX

A disciplina permitiu demonstrar a integracao fora do APEX nesta entrega. Por isso, o foco tecnico ficou em IA funcional, API REST consumivel e painel demonstrativo.
