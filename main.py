from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import whisper
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import tempfile, os, io

app = FastAPI()

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# Load Whisper once (faster reuse)
model = whisper.load_model("base")

@app.get("/")
def root():
    return {"message": "Universal Speech Translator API with Whisper (Auto-detect speech language)"}


@app.post("/speech_translate")
async def speech_translate(
    file: UploadFile = File(...),
    target_lang: str = Query(..., description="Target language code (e.g., 'en', 'ta', 'fr')"),
    format: str = Query("mp3", enum=["mp3", "wav", "ogg"])
):
    """
    Upload speech (any language), auto-detect with Whisper,
    translate into target_lang, and return translated audio.
    """

    try:
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        # Step 1: Transcribe & detect language with Whisper
        try:
            result = model.transcribe(temp_path)
            text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {str(e)}")

        # Step 2: Translate text
        try:
            translated_text = GoogleTranslator(source="auto", target=target_lang).translate(text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

        # Step 3: Text → Speech (gTTS → MP3)
        mp3_bytes = io.BytesIO()
        try:
            tts = gTTS(text=translated_text, lang=target_lang)
            tts.write_to_fp(mp3_bytes)
            mp3_bytes.seek(0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

        # Step 4: Convert MP3 → requested format
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
        if format == "mp3":
            with open(out_file.name, "wb") as f:
                f.write(mp3_bytes.read())
        else:
            sound = AudioSegment.from_file(io.BytesIO(mp3_bytes.read()), format="mp3")
            sound.export(out_file.name, format=format)

        return JSONResponse({
            "detected_language": detected_lang,
            "original_text": text,
            "translated_text": translated_text,
            "download_url": f"/download_audio?path={out_file.name}"
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/download_audio")
def download_audio(path: str):
    """Serve generated audio file"""
    if os.path.exists(path):
        ext = path.split(".")[-1]
        return FileResponse(path, media_type=f"audio/{ext}", filename=f"output.{ext}")
    return {"error": "Audio file not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
