import streamlit as st
import numpy as np
import faiss

from openai import OpenAI, RateLimitError
from utils import (
    extract_text_from_pdf,
    split_text,
    embed_chunks,
    search_similar
)

# =====================================================
# Configuração da página
# =====================================================

st.set_page_config(
    page_title="NutriBot ADA 2019",
    layout="wide"
)

st.title("🤖 NutriBot – baseado no Consenso ADA 2019")

st.info(
    "Este sistema responde com base no Consenso ADA 2019. "
    "As informações têm finalidade educacional e não substituem "
    "consulta com médico, nutricionista ou outro profissional de saúde."
)

# =====================================================
# Inicialização segura do cliente OpenAI
# =====================================================

def get_openai_client():
    """
    Busca a chave da OpenAI nos Secrets do Streamlit.

    Aceita dois formatos:

    1) Formato com seção:
       [openai]
       api_key = "SUA_CHAVE"

    2) Formato simples:
       OPENAI_API_KEY = "SUA_CHAVE"
    """

    api_key = None

    try:
        if "openai" in st.secrets:
            api_key = st.secrets["openai"].get("api_key", None)

        if api_key is None and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]

    except Exception:
        api_key = None

    if not api_key:
        st.error(
            "Chave da OpenAI não encontrada.\n\n"
            "Configure os Secrets do Streamlit usando um dos formatos abaixo:\n\n"
            "Formato 1:\n"
            "[openai]\n"
            "api_key = \"SUA_CHAVE_DA_OPENAI\"\n\n"
            "ou\n\n"
            "Formato 2:\n"
            "OPENAI_API_KEY = \"SUA_CHAVE_DA_OPENAI\""
        )
        st.stop()

    return OpenAI(api_key=api_key)


client = get_openai_client()

# =====================================================
# Preparação do índice vetorial com FAISS
# =====================================================

@st.cache_resource(show_spinner="🔍 Processando o Consenso ADA 2019...")
def prepare_index(_client):
    pdf_path = "CONSENSO ADA 2019.pdf"

    texto = extract_text_from_pdf(pdf_path)

    if not texto or len(texto.strip()) == 0:
        raise ValueError(
            "Não foi possível extrair texto do PDF. "
            "Verifique se o arquivo existe e se contém texto selecionável."
        )

    chunks = split_text(texto, max_tokens=500)

    if not chunks:
        raise ValueError(
            "Nenhum trecho foi gerado a partir do PDF."
        )

    embeddings, clean_chunks = embed_chunks(chunks, _client)

    if not embeddings:
        raise ValueError(
            "Nenhum embedding foi gerado. Verifique a função embed_chunks."
        )

    embeddings = np.array(embeddings).astype("float32")

    if embeddings.ndim != 2:
        raise ValueError(
            "Os embeddings precisam estar no formato de matriz 2D."
        )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, clean_chunks


try:
    index, chunks = prepare_index(client)
except FileNotFoundError:
    st.error(
        "Arquivo PDF não encontrado. Verifique se o arquivo "
        "'CONSENSO ADA 2019.pdf' está na mesma pasta do main.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Erro ao preparar o índice vetorial: {e}")
    st.stop()

# =====================================================
# Interface do usuário
# =====================================================

st.markdown("### Faça uma pergunta")

user_input = st.text_area(
    "Digite sua pergunta sobre diabetes:",
    placeholder="Exemplo: Quais são as recomendações nutricionais para pessoas com diabetes tipo 2?"
)

top_k = st.slider(
    "Número de trechos do documento usados como contexto:",
    min_value=2,
    max_value=8,
    value=4
)

# =====================================================
# Função para gerar resposta
# =====================================================

def responder_pergunta(pergunta, top_k=4):
    contextos = search_similar(
        pergunta,
        index,
        chunks,
        client,
        top_k=top_k
    )

    if not contextos:
        return (
            "Não encontrei trechos relevantes no Consenso ADA 2019 para responder à pergunta.",
            []
        )

    contexto_completo = "\n\n---\n\n".join(contextos)

    system_prompt = f"""
Você é o NutriBot, um assistente educacional baseado exclusivamente no Consenso ADA 2019.

Regras obrigatórias:
1. Responda apenas com base no CONTEXTO fornecido.
2. Se a resposta não estiver no contexto, diga:
   "Não encontrei essa informação no Consenso ADA 2019 usado como base."
3. Não invente informações.
4. Não forneça diagnóstico individual.
5. Não prescreva tratamento individualizado.
6. Oriente o usuário a procurar um profissional de saúde quando a pergunta envolver conduta clínica individual.
7. Responda em português, com linguagem clara e objetiva.

CONTEXTO:
{contexto_completo}
"""

    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pergunta}
        ]
    )

    texto_resposta = resposta.choices[0].message.content

    return texto_resposta, contextos

# =====================================================
# Botão principal
# =====================================================

if st.button("📤 Perguntar ao NutriBot"):
    pergunta = user_input.strip()

    if not pergunta:
        st.warning("Por favor, insira uma pergunta válida.")
    else:
        try:
            resposta, contextos_usados = responder_pergunta(
                pergunta,
                top_k=top_k
            )

            st.markdown("### 💬 Resposta baseada no Consenso ADA 2019")
            st.markdown(resposta)

            if contextos_usados:
                with st.expander("📚 Ver trechos do documento usados como contexto"):
                    for i, trecho in enumerate(contextos_usados, start=1):
                        st.markdown(f"**Trecho {i}:**")
                        st.write(trecho)

        except RateLimitError:
            st.error(
                "🚫 Limite de requisições atingido. "
                "Aguarde um pouco e tente novamente."
            )

        except Exception as e:
            st.error(f"⚠️ Ocorreu um erro inesperado: {e}")




