import streamlit as st
import numpy as np
import faiss

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from openai import OpenAI, RateLimitError


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
    "As informações têm finalidade educacional e não substituem consulta "
    "com médico, nutricionista ou outro profissional de saúde."
)


# =====================================================
# Cliente OpenAI
# =====================================================

def get_openai_client():
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
            "Configure os Secrets do Streamlit Cloud usando um dos formatos abaixo:\n\n"
            "[openai]\n"
            "api_key = \"SUA_CHAVE_DA_OPENAI\"\n\n"
            "ou\n\n"
            "OPENAI_API_KEY = \"SUA_CHAVE_DA_OPENAI\""
        )
        st.stop()

    return OpenAI(api_key=api_key)


client = get_openai_client()


# =====================================================
# Extração de texto do PDF
# =====================================================

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# =====================================================
# Divisão do texto em trechos
# =====================================================

def split_text(text, max_chars=1800, overlap=250):
    text = text.replace("\n", " ")
    words = text.split()

    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chars:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:]
            current_chunk = overlap_text + " " + word

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# =====================================================
# Geração de embeddings com OpenAI
# =====================================================

def embed_texts(texts, client, batch_size=50):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings).astype("float32")


def embed_query(query, client):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )

    embedding = np.array(
        [response.data[0].embedding]
    ).astype("float32")

    return embedding


# =====================================================
# Preparação do índice FAISS
# =====================================================

@st.cache_resource(show_spinner="🔍 Processando o Consenso ADA 2019 com OpenAI...")
def prepare_index(_client):
    pdf_path = "CONSENSO ADA 2019.pdf"

    texto = extract_text_from_pdf(pdf_path)

    if not texto or len(texto.strip()) == 0:
        raise ValueError(
            "Não foi possível extrair texto do PDF. "
            "Verifique se o arquivo contém texto selecionável."
        )

    chunks = split_text(texto)

    if not chunks:
        raise ValueError("Nenhum trecho foi gerado a partir do PDF.")

    embeddings = embed_texts(chunks, _client)

    if embeddings is None or embeddings.shape[0] == 0:
        raise ValueError("Nenhum embedding foi gerado.")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


# =====================================================
# Busca semântica
# =====================================================

def search_similar(query, index, chunks, client, top_k=4):
    query_embedding = embed_query(query, client)

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append(
                {
                    "score": float(score),
                    "text": chunks[idx]
                }
            )

    return results


# =====================================================
# Geração da resposta com ChatGPT
# =====================================================

def generate_answer(question, contexts, client):
    contexto_completo = "\n\n---\n\n".join(
        [item["text"] for item in contexts]
    )

    system_prompt = f"""
Você é o NutriBot, um assistente educacional baseado exclusivamente no Consenso ADA 2019.

Regras obrigatórias:
1. Responda apenas com base no CONTEXTO fornecido.
2. Se a resposta não estiver no CONTEXTO, diga:
   "Não encontrei essa informação no Consenso ADA 2019 usado como base."
3. Não invente informações.
4. Não forneça diagnóstico individual.
5. Não prescreva tratamento individualizado.
6. Quando a pergunta envolver conduta clínica individual, oriente o usuário a procurar um profissional de saúde.
7. Responda em português, com linguagem clara, objetiva e cuidadosa.

CONTEXTO:
{contexto_completo}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content


# =====================================================
# Inicialização do índice
# =====================================================

try:
    index, chunks = prepare_index(client)

except FileNotFoundError:
    st.error(
        "Arquivo PDF não encontrado. Verifique se o arquivo "
        "'CONSENSO ADA 2019.pdf' está na mesma pasta do main.py."
    )
    st.stop()

except RateLimitError:
    st.error(
        "A API da OpenAI retornou erro de limite ou cota.\n\n"
        "Verifique se sua conta da API possui créditos disponíveis. "
        "O ChatGPT Plus não é a mesma coisa que crédito de API."
    )
    st.stop()

except Exception as e:
    erro = str(e)

    if "insufficient_quota" in erro or "exceeded your current quota" in erro:
        st.error(
            "Sua chave da OpenAI está correta, mas a conta está sem cota/crédito na API.\n\n"
            "Para usar o app via ChatGPT, você precisa ativar billing ou adicionar créditos "
            "na plataforma da OpenAI."
        )
    else:
        st.error(f"Erro ao preparar o índice vetorial: {e}")

    st.stop()


# =====================================================
# Interface
# =====================================================

st.markdown("### Faça uma pergunta")

user_input = st.text_area(
    "Digite sua pergunta sobre diabetes:",
    placeholder=(
        "Exemplo: Quais são as recomendações nutricionais "
        "para pessoas com diabetes tipo 2?"
    )
)

top_k = st.slider(
    "Número de trechos usados como contexto:",
    min_value=2,
    max_value=10,
    value=4
)


# =====================================================
# Execução
# =====================================================

if st.button("📤 Perguntar ao NutriBot"):
    pergunta = user_input.strip()

    if not pergunta:
        st.warning("Por favor, insira uma pergunta válida.")

    else:
        try:
            with st.spinner("Buscando trechos relevantes no Consenso ADA 2019..."):
                contextos = search_similar(
                    pergunta,
                    index,
                    chunks,
                    client,
                    top_k=top_k
                )

            if not contextos:
                st.warning(
                    "Não encontrei trechos relevantes no documento para essa pergunta."
                )

            else:
                with st.spinner("Gerando resposta com ChatGPT..."):
                    resposta = generate_answer(
                        pergunta,
                        contextos,
                        client
                    )

                st.markdown("### 💬 Resposta baseada no Consenso ADA 2019")
                st.markdown(resposta)

                with st.expander("📚 Ver trechos usados como contexto"):
                    for i, item in enumerate(contextos, start=1):
                        st.markdown(f"#### Trecho {i}")
                        st.caption(f"Similaridade: {item['score']:.3f}")
                        st.write(item["text"])
                        st.divider()

        except RateLimitError:
            st.error(
                "Erro de limite/cota da API OpenAI. "
                "Verifique seus créditos e limites no painel da OpenAI."
            )

        except Exception as e:
            erro = str(e)

            if "insufficient_quota" in erro or "exceeded your current quota" in erro:
                st.error(
                    "Sua conta da API OpenAI está sem cota/crédito. "
                    "Para usar via ChatGPT, será necessário adicionar créditos no billing da OpenAI."
                )
            else:
                st.error(f"Erro ao processar a pergunta: {e}")
