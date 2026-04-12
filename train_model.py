import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def gerar_dados_mock(qtd_alunos=1000):
    np.random.seed(42)
    
    # Criando colunas que impactam na desistência
    horas_estudadas = np.random.randint(2, 60, size=qtd_alunos)
    exercicios_concluidos = np.random.randint(0, 20, size=qtd_alunos)
    media_notas = np.random.uniform(3.0, 10.0, size=qtd_alunos)
    dias_inativos = np.random.randint(0, 30, size=qtd_alunos)
    
    # Lógica de evasão (quanto menos estudo/exercicios e mais inativo, maior chance)
    prob_evasao = (dias_inativos * 0.4) - (horas_estudadas * 0.2) - (exercicios_concluidos * 0.3) - (media_notas * 0.1)
    
    # Normalizando para probabilidades e convertendo em classes
    limite = np.percentile(prob_evasao, 70) # Forçando uns 30% de evasão
    evasao = (prob_evasao > limite).astype(int)

    df = pd.DataFrame({
        'horas_estudadas': horas_estudadas,
        'exercicios_concluidos': exercicios_concluidos,
        'media_notas': media_notas,
        'dias_inativos': dias_inativos,
        'evasao': evasao
    })
    return df

print("1. Gerando dados de treinamento...")
df = gerar_dados_mock(2500)

X = df[['horas_estudadas', 'exercicios_concluidos', 'media_notas', 'dias_inativos']]
y = df['evasao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Treinando o Modelo Random Forest...")
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"-> Acurácia do modelo no teste: {score:.2%}")

print("3. Salvando o modelo na raiz do projeto (modelo_evasao.pkl)...")
with open("modelo_evasao.pkl", "wb") as f:
    pickle.dump(clf, f)
    
print("Pronto! Modelo criado com sucesso. Agora você pode rodar o Streamlit.")
