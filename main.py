import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =====================================================
# Configuração da página
# =====================================================

st.set_page_config(
    page_title="NutriBot ADA 2019",
    layout="wide"
)

st.title("🤖 NutriBot – baseado no Consenso ADA 2019")

st.info(
    "Este sistema realiza busca semântica no Consenso ADA 2019. "
    "As informações têm finalidade educacional e não substituem consulta "
    "com médico, nutricionista ou outro profissional de saúde."
)


# =====================================================
# Função para extrair texto do PDF
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
# Função para dividir o texto em trechos
# =====================================================

def split_text(text, max_chars=1800, overlap=250):
    """
    Divide o texto em blocos parcialmente sobrepostos.

    max_chars: tamanho aproximado de cada trecho.
    overlap: número aproximado de caracteres repetidos entre trechos.
    """

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
# Carregamento do modelo local de embeddings
# =====================================================

@st.cache_resource(show_spinner="Carregando modelo local de embeddings...")
def load_model():
    """
    Modelo multilíngue leve.
    Funciona bem para português e não depende da API da OpenAI.
    """
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return model


# =====================================================
# Preparação do índice vetorial com FAISS
# =====================================================

@st.cache_resource(show_spinner="🔍 Processando o Consenso ADA 2019...")
def prepare_index():
    pdf_path = "CONSENSO ADA 2019.pdf"

    texto = extract_text_from_pdf(pdf_path)

    if not texto or len(texto.strip()) == 0:
        raise ValueError(
            "Não foi possível extrair texto do PDF. "
            "Verifique se o arquivo contém texto selecionável."
        )

    chunks = split_text(texto)

    if not chunks:
        raise ValueError(
            "Nenhum trecho foi gerado a partir do PDF."
        )

    model = load_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Correção do erro:
    # "The truth value of an array with more than one element is ambiguous"
    if embeddings is None or embeddings.shape[0] == 0:
        raise ValueError(
            "Nenhum embedding foi gerado."
        )

    embeddings = embeddings.astype("float32")

    # Normaliza os vetores para usar similaridade por produto interno
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks, model


# =====================================================
# Busca semântica
# =====================================================

def search_similar(query, index, chunks, model, top_k=4):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    if query_embedding is None or query_embedding.shape[0] == 0:
        raise ValueError(
            "Não foi possível gerar embedding para a pergunta."
        )

    query_embedding = query_embedding.astype("float32")
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
# Inicialização do índice
# =====================================================

try:
    index, chunks, model = prepare_index()

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
    placeholder=(
        "Exemplo: Quais são as recomendações nutricionais "
        "para pessoas com diabetes tipo 2?"
    )
)

top_k = st.slider(
    "Número de trechos recuperados do documento:",
    min_value=2,
    max_value=10,
    value=4
)


# =====================================================
# Execução da busca
# =====================================================

if st.button("📤 Perguntar ao NutriBot"):
    pergunta = user_input.strip()

    if not pergunta:
        st.warning("Por favor, insira uma pergunta válida.")

    else:
        try:
            resultados = search_similar(
                pergunta,
                index,
                chunks,
                model,
                top_k=top_k
            )

            if not resultados:
                st.warning(
                    "Não encontrei trechos relevantes no documento para essa pergunta."
                )

            else:
                st.markdown(
                    "### 📚 Trechos mais relevantes encontrados no Consenso ADA 2019"
                )

                for i, resultado in enumerate(resultados, start=1):
                    st.markdown(f"#### Trecho {i}")
                    st.caption(f"Similaridade: {resultado['score']:.3f}")
                    st.write(resultado["text"])
                    st.divider()

                st.warning(
                    "Nesta versão sem OpenAI, o sistema recupera os trechos "
                    "mais relevantes, mas ainda não gera uma resposta sintetizada "
                    "automaticamente."
                )

        except Exception as e:
            st.error(f"Erro ao processar a pergunta: {e}")
