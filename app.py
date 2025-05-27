import streamlit as st
import openai

# Updated for OpenAI >= v1.0
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_post(idea, voice_description):
    prompt = f"""
    You are an expert LinkedIn content writer who writes posts in a personal, insightful voice. The user's tone of voice is described as: {voice_description}.

    Based on the following idea, write a short LinkedIn post that feels authentic and engaging. Avoid sounding too formal or too casual. Use formatting like short paragraphs, line breaks, or emojis if it suits the voice.

    Idea: {idea}

    Write the post:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("LinkedIn Post Generator")

idea = st.text_input("Enter your post idea:")
voice_description = st.text_input("Describe your voice (e.g., optimistic, thoughtful):")

if st.button("Generate Post"):
    if idea and voice_description:
        post = generate_post(idea, voice_description)
        st.write("### Generated LinkedIn Post:")
        st.write(post)
    else:
        st.warning("Please provide both the idea and voice description.")

