from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.utils import transcribe_audio, ask_gpt, synthesize_speech
import os
import tempfile
import base64

app = FastAPI()

@app.post("/process-voice")
async def process_voice(file: UploadFile = File(...)):
    # Sauvegarde temporaire de l'audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    # Étapes : audio -> texte -> réponse -> audio
    transcription = transcribe_audio(temp_path)
    print("🎙️ Transcrit :", transcription)
    reply = ask_gpt(transcription)
    print("🤖 GPT :", reply)
    mp3_path = synthesize_speech(reply)

    os.remove(temp_path)

    # Encode le fichier MP3 en base64
    with open(mp3_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    os.remove(mp3_path)

    return JSONResponse(content={
        "transcript": transcription,  # ✅ Ajouté : ce que l'utilisateur a dit
        "response": reply,            # ✅ Réponse GPT
        "audio": audio_base64         # ✅ Audio encodé
    })
