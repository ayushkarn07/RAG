from groq import Groq
from config import GROQ_MODEL

client = Groq()  # automatically reads GROQ_API_KEY from env

def call_groq(prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_tokens
    )
    return completion.choices[0].message.content
