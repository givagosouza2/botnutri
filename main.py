import streamlit as st
import openai
import os
from utils import extract_text_from_pdf, split_text, embed_chunks, search_similar
import faiss

st.set_page_config(page_title="NutriBot ADA 2019", layout="wide")
st.title("🤖 NutriBot – baseado no Consenso ADA 2019")

openai.api_key = st.secrets["openai"]["api_key"]

# Carregamento e indexação
@st.cache_resource(show_spinner="Processando o PDF...")
def prepare_index():
    text = extract_text_from_pdf("CONSENSO ADA 2019.pdf")
    chunks = split_text(text, max_tokens=500)
    vectors, clean_chunks = embed_chunks(chunks)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)
    return index, clean_chunks

index, chunks = prepare_index()

user_input = st.text_input("Digite sua pergunta sobre diabetes:")

if st.button("📤 Perguntar ao NutriBot"):
    if user_input.strip() == "":
        st.warning("Por favor, digite uma pergunta.")
    else:
        # Etapa 1: verificar se é sobre diabetes
        filtro = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Responda apenas com SIM ou NÃO. A seguinte pergunta está relacionada ao tema diabetes?"},
                {"role": "user", "content": user_input}
            ]
        )
        if "SIM" in filtro.choices[0].message.content.upper():
            # Etapa 2: busca nos trechos relevantes do PDF
            contextos = search_similar(user_input, index, chunks, top_k=4)
            contexto_unido = "\n\n".join(contextos)

            resposta = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente que responde apenas com base no conteúdo fornecido. Responda à pergunta com base no seguinte contexto extraído do Consenso ADA 2019:\n\n" + contexto_unido},
                    {"role": "user", "content": user_input}
                ]
            )

            st.markdown("💬 **Resposta:**")
            st.markdown(resposta.choices[0].message.content)
        else:
            st.warning("❌ O NutriBot responde apenas a perguntas sobre **diabetes** com base no conteúdo do Consenso ADA 2019.")
