import streamlit as st
from openai import OpenAI
from openai import RateLimitError


st.set_page_config(
    page_title="NutriBot",
    page_icon="😃",
    layout="wide"
)

# Configura a chave da API
openai.api_key = st.secrets["openai"]["api_key"]

st.title("NutriBot")

user_text = st.text_input("Inserir texto")

if st.button("📤 Enviar avaliação ao ChatGPT"):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_text}]
        )

        st.markdown(response['choices'][0]['message']['content'])

    except RateLimitError:
        st.error("❌ Você atingiu o limite de requisições. Tente novamente em instantes.")
    except Exception as e:
        st.error(f"❌ Ocorreu um erro: {str(e)}")
