import streamlit as st
from openai import OpenAI
from openai import RateLimitError

st.set_page_config(
    page_title="NutriBot",
    page_icon="😃",
    layout="wide"
)

# Instancia o cliente
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

st.title("NutriBot")
user_text = st.text_input("Inserir sua pergunta sobre diabetes")

if st.button("📤 Enviar avaliação ao ChatGPT"):
    try:
        # Passo 1: Verifica se a pergunta é sobre diabetes
        check_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Sua tarefa é verificar se a pergunta está relacionada ao tema 'diabetes'. Responda apenas com 'SIM' ou 'NÃO'."
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        )

        is_about_diabetes = check_response.choices[0].message.content.strip().upper()

        if "SIM" in is_about_diabetes:
            # Passo 2: Pergunta validada — agora responde normalmente
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente nutricional especializado em diabetes. Responda de forma clara e com base em evidências."},
                    {"role": "user", "content": user_text}
                ]
            )
            st.markdown(response.choices[0].message.content)

        else:
            # Pergunta fora do escopo
            st.warning("❌ O NutriBot responde apenas a perguntas relacionadas ao diabetes. Reformule sua pergunta dentro desse tema.")

    except RateLimitError:
        st.error("❌ Você atingiu o limite de requisições. Tente novamente em instantes.")
    except Exception as e:
        st.error(f"❌ Ocorreu um erro: {str(e)}")
