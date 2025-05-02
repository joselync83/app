import gradio as gr
from transformers import MarianMTModel, MarianTokenizer
import openai
import os

# Cargar clave desde variable de entorno
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Traducción
model_name = 'Helsinki-NLP/opus-mt-es-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def traducir(texto):
    inputs = tokenizer(texto, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def mejorar(texto_en):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Reescribe este texto para que suene más profesional y claro:\n\n\"{texto_en}\""}],
        temperature=0.4,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def flujo_completo(texto):
    traduccion = traducir(texto)
    return mejorar(traduccion)

interface = gr.Interface(
    fn=flujo_completo,
    inputs=gr.Textbox(lines=4, placeholder="Escribe en español...", label="Texto original"),
    outputs=gr.Textbox(label="Texto profesional en inglés"),
    title="Traductor Profesional con IA",
    description="Convierte textos en inglés claro y profesional automáticamente",
    theme="soft"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)