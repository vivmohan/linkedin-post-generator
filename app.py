import openai
import faiss
import streamlit as st
import numpy as np

# OpenAI key from secrets
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Sample posts (you can expand this later)
past_posts = [
    "Excited to be with the SoundHealth team at the Osher Integrative Medicine Conference...",
    "SONUCast is the first allergy predictor that uses facial anatomy and the environment around you...",
    "I am thrilled to share news about the launch of SONU and the closing of SoundHealth's $7 million seed funding round!",
    "Quality sleep is more essential than ever. Our pilot study with Stanford Medicine showed acoustic resonance therapy...",
    "Thanks Amit Garg and Tau Ventures for being a great partner to SoundHealth...",
]

# Create FAISS index
embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)
embedded_posts = []

# Embed and index each post
for post in past_posts:
    response = client.embeddings.create(input=post, model="text-embedding-3-large")
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    embedded_posts.append((post, vector))
    index.add(np.array([vector]))

def retrieve_similar_posts(query, k=3):
    response = client.embeddings.create(input=query, model="text-embedding-3-large")
    vector = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    D, I = index.search(vector, k)
    return [embedded_posts[i][0] for i in I[0]]

def generate_post(idea, similar_posts):
    prompt = f"""
You are writing a LinkedIn post in the voice of this user, based on their past writing style:

{chr(10).join(similar_posts)}

Now write a new post on: "{idea}"

Match the tone: thoughtful, credible, research-driven. Avoid marketing hype. Start with an insight, support with science or product detail, end with a soft call to action.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("LinkedIn Post Generator (In Your Voice)")
idea = st.text_input("Enter your post idea:")

if st.button("Generate Post"):
    if idea:
        examples = retrieve_similar_posts(idea)
        post = generate_post(idea, examples)
        st.write("### Generated Post:")
        st.write(post)
    else:
        st.warning("Please enter a post idea.")

