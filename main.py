import streamlit as st
import os
from openai import OpenAI, RateLimitError
from utils import extract_text_from_pdf, split_text, embed_chunks, search_similar
import faiss

# Inicializa API
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
st.write("🔐 API Key carregada:", st.secrets["openai"]["api_key"][:5], "********")
st.set_page_config(page_title="NutriBot ADA 2019", layout="wide")
st.title("🤖 NutriBot – baseado no Consenso ADA 2019")

# Carregamento do PDF e criação de índice
@st.cache_resource(show_spinner="🔍 Processando o Consenso ADA 2019...")
def prepare_index():
    texto = extract_text_from_pdf("CONSENSO ADA 2019.pdf")
    chunks = split_text(texto, max_tokens=500)
    embeddings, clean_chunks = embed_chunks(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, clean_chunks

index, chunks = prepare_index()

# Entrada do usuário
user_input = st.text_input("Digite sua pergunta sobre diabetes:")

if st.button("📤 Perguntar ao NutriBot"):
    if not user_input.strip():
        st.warning("Por favor, insira uma pergunta válida.")
    else:
        try:
            # Passo 1: Verifica se é sobre diabetes
            filtro = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Responda apenas com 'SIM' ou 'NÃO'. A pergunta a seguir está relacionada ao tema diabetes?"},
                    {"role": "user", "content": user_input}
                ]
            )
            is_diabetes = filtro.choices[0].message.content.strip().upper()

            if "SIM" in is_diabetes:
                # Passo 2: Busca nos trechos relevantes
                contextos = search_similar(user_input, index, chunks, top_k=4)
                contexto_completo = "\n\n".join(contextos)

                resposta = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"Você é um assistente especializado que responde exclusivamente com base no conteúdo fornecido do Consenso ADA 2019. Use apenas as informações abaixo para responder:\n\n{contexto_completo}"},
                        {"role": "user", "content": user_input}
                    ]
                )

                st.markdown("💬 **Resposta baseada no Consenso ADA 2019:**")
                st.markdown(resposta.choices[0].message.content)

            else:
                st.warning("❌ O NutriBot responde apenas a perguntas sobre **diabetes** com base no conteúdo do Consenso ADA 2019.")
        except RateLimitError:
            st.error("🚫 Limite de requisições atingido. Aguarde um momento e tente novamente.")
        except Exception as e:
            st.error(f"⚠️ Erro inesperado: {str(e)}")
