from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from app.utils import transcribe_audio, ask_gpt, synthesize_speech
import os
import tempfile

app = FastAPI()

@app.post("/process-voice")
async def process_voice(file: UploadFile = File(...)):
    # Sauvegarde temporaire de l'audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    # Étapes de traitement
    text = transcribe_audio(temp_path)
    print("🎙️ Transcrit :", text)
    reply = ask_gpt(text)
    print("🤖 GPT :", reply)
    mp3_path = synthesize_speech(reply)

    # Nettoyage du fichier source
    os.remove(temp_path)

    return FileResponse(mp3_path, media_type="audio/mpeg", filename="response.mp3")
