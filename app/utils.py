import tempfile
import os
import time
import requests
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🧠 Assistant ID global
ASSISTANT_ID = None

# 🔍 Brave API key
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# 🔧 Fonction externe accessible à GPT (Function Calling)
search_web_function = {
    "name": "search_web",
    "description": "Effectue une recherche web avec Brave Search pour obtenir des informations actuelles.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "La question ou sujet à rechercher."
            }
        },
        "required": ["query"]
    }
}

# 🔍 Requête à Brave Search
def search_web(query: str) -> str:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {"q": query, "count": 3}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json().get("web", {}).get("results", [])
        if not results:
            return "Aucun résultat trouvé."
        return "\n\n".join([f"{r['title']} - {r['url']}\n{r['description']}" for r in results])
    return "Erreur lors de la recherche web."

# 📌 Assistant simple avec Function Calling
conversation = [
    {"role": "system", "content": "Tu es Alto, un assistant vocal intelligent et connecté. Utilise les fonctions externes si nécessaire."}
]

async def ask_gpt(prompt):
    conversation.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
        functions=[search_web_function],
        function_call="auto"
    )

    message = response.choices[0].message

    if message.function_call:
        name = message.function_call.name
        args = eval(message.function_call.arguments)

        if name == "search_web":
            result = search_web(args["query"])
            conversation.append({
                "role": "function",
                "name": name,
                "content": f"Voici les résultats web trouvés :\n{result}\n\nMerci de résumer l'information principale trouvée dans un langage naturel, comme si tu parlais à un humain."
            })

            followup = await client.chat.completions.create(
                model="gpt-4o",
                messages=conversation
            )
            answer = followup.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": answer})
            return answer

    answer = message.content.strip()
    conversation.append({"role": "assistant", "content": answer})
    return answer

# 🎤 Transcription audio
async def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

# 🎧 Synthèse vocale
async def synthesize_speech(text):
    input_text = "Hum......... " + text
    speech = await client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=input_text
    )
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.write(speech.content)
    return temp_file.name
