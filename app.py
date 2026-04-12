import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Dashboard IA - Evasão Escolar", page_icon="🎓", layout="wide")

# Cabecalho
st.title("🎓 Sistema de Predição de Evasão (Mock APEX)")
st.write("Este dashboard simula um ambiente administrativo corporativo Oracle APEX utilizando o nosso algoritmo de IA nativo para analisar e identificar alunos com risco de abandonar o curso da plataforma.")

st.divider()

# Carregar Modelo Pre-treinado
@st.cache_resource
def load_model():
    try:
        with open("modelo_evasao.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

modelo = load_model()

if not modelo:
    st.error("⚠️ Modelo não encontrado! Execute 'python train_model.py' primeiro para gerar o arquivo 'modelo_evasao.pkl'.")
    st.stop()

# Layout Principal em duas colunas
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simular Análise Individual")
    with st.form("form_analise"):
        st.write("Insira os dados da plataforma educacional:")
        horas_estudadas = st.slider("Horas Estudadas", 0, 100, 20)
        exercicios_concluidos = st.slider("Exercícios Concluídos", 0, 50, 10)
        media_notas = st.slider("Média de Notas (0-10)", 0.0, 10.0, 7.5, 0.1)
        dias_inativos = st.slider("Dias Consecutivos Inativo", 0, 90, 5)
        
        analisar_btn = st.form_submit_button("Consultar Algoritmo de IA")

with col2:
    st.subheader("Resultado Predição (IA)")
    if analisar_btn:
        df_input = pd.DataFrame([{
            "horas_estudadas": horas_estudadas,
            "exercicios_concluidos": exercicios_concluidos,
            "media_notas": media_notas,
            "dias_inativos": dias_inativos
        }])
        
        # Realizando Predição
        predicao = modelo.predict(df_input)[0]
        probabilidades = modelo.predict_proba(df_input)[0]
        prob_evasao_percent = probabilidades[1] * 100
        
        if predicao == 1:
            st.error(f"🚨 **ALERTA CRÍTICO**")
            st.write(f"Risco de Evasão/Desistência Detectado! (A probabilidade calculada foi de {prob_evasao_percent:.1f}%)")
            st.write("→ **Sugestão de Ação Automática:** Acionar a equipe de suporte ao estudante ou enviar e-mail automático oferecendo monitoria/ajuda com as tarefas pendentes.")
        else:
            st.success(f"✅ **NORMAL**")
            st.write(f"Nenhum risco de evasão proeminente detectado no momento. (Probabilidade de {prob_evasao_percent:.1f}%)")
            st.write("→ **Sugestão de Ação:** Prosseguir com o fluxo de vida diária letiva do aluno.")
            
    else:
        st.info("Preencha os dados do aluno no menu lateral e clique em Consultar Algoritmo de IA para ver a simulação da nossa rede inferindo o dropout.")

st.divider()
st.subheader("Análise em Lote (Demonstração do Relatório APEX)")
st.write("Abaixo vemos uma amostra simulada de tabelas Oracle Database sendo processadas e listadas pela IA de uma só vez.")

# Dados falsos na tabela
dados_mock = pd.DataFrame({
    'Nome Aluno': ['João Silva', 'Maria Souza', 'Carlos Eduardo', 'Ana Beatriz', 'José Fernandes'],
    'horas_estudadas': [15, 80, 5, 45, 12],
    'exercicios_concluidos': [5, 40, 1, 20, 2],
    'media_notas': [5.5, 9.8, 3.0, 8.5, 4.0],
    'dias_inativos': [10, 0, 25, 2, 30]
})

predicoes_batch = modelo.predict(dados_mock.drop(columns=['Nome Aluno']))
dados_mock['Status IA (Predição)'] = ['🚨 Risco de Evasão!' if p == 1 else '✅ Regular' for p in predicoes_batch]

st.dataframe(dados_mock, use_container_width=True)
