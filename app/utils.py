import tempfile
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")  # 🔐 stocke ton Assistant ID dans Render

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

def query_assistant(user_input: str) -> str:
    # Crée un thread
    thread = client.beta.threads.create()

    # Ajoute le message de l'utilisateur au thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    # Lance l'exécution de l'assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    )

    # Attend que l'assistant réponde
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status in ["failed", "cancelled"]:
            return "Erreur : l'assistant n'a pas pu répondre."
        time.sleep(1)

    # Récupère la réponse la plus récente
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for message in reversed(messages.data):
        if message.role == "assistant":
            return message.content[0].text.value.strip()

    return "Erreur : aucune réponse reçue."

def synthesize_speech(text):
    input_text = "Hum... " + text
    speech = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=input_text
    )
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.write(speech.content)
    return temp_file.name
