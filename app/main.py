from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import base64
from app.utils import transcribe_audio, query_assistant, synthesize_speech

app = FastAPI()

@app.post("/process-voice")
async def process_voice(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    user_transcript = transcribe_audio(temp_path)
    print("🎙️ Transcrit :", user_transcript)
    assistant_reply = query_assistant(user_transcript)
    print("🤖 Assistant :", assistant_reply)
    mp3_path = synthesize_speech(assistant_reply)

    os.remove(temp_path)

    with open(mp3_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(mp3_path)

    return JSONResponse(content={
        "transcript": user_transcript,
        "response": assistant_reply,
        "audio": audio_base64
    })
