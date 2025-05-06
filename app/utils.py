import tempfile
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conversation = [
    {"role": "system", "content": "Tu es un assistant vocal amical et intelligent."}
]

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

def ask_gpt(prompt):
    conversation.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation
    )
    message = response.choices[0].message.content.strip()
    conversation.append({"role": "assistant", "content": message})
    return message

def synthesize_speech(text):
    input_text = "Hum......... " + text 
    speech = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=input_text
    )
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.write(speech.content)
    return temp_file.name
