import streamlit as st
import pandas as pd
from openai import OpenAI
from openai import RateLimitError

st.set_page_config(
    page_title="NutriBot",
    page_icon="😃",
    layout="wide"
)
# Cria cliente com a chave da API
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

st.title("NutriBot")
input = st.text_input("inserir texto")
if st.button("📤 Enviar avaliação ao ChatGPT"):
    try:
        user_input = input

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )

        st.markdown(response.choices[0].message.content)

    except RateLimitError:
        st.error("❌ Você atingiu o limite de requisições. Tente novamente em instantes.")
    except Exception as e:
        st.error(f"❌ Ocorreu um erro: {str(e)}")
