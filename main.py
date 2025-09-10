from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import tempfile
import os
from pydub import AudioSegment  # for converting formats

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hindi → Tamil Speech Translator API"}


@app.post("/speech_translate")
async def speech_translate(
    file: UploadFile = File(...),
    format: str = Query("mp3", enum=["mp3", "wav", "ogg"])
):
    """
    Upload Hindi speech (.wav), translate to Tamil, and return Tamil speech in desired format.
    """

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported for input.")

    try:
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        # Step 1: Speech to Hindi text
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            try:
                hindi_text = recognizer.recognize_google(audio, language="hi-IN")
            except sr.UnknownValueError:
                raise HTTPException(status_code=400, detail="Could not understand the Hindi speech.")
            except sr.RequestError:
                raise HTTPException(status_code=503, detail="Speech recognition service unavailable.")

        # Step 2: Translate Hindi → Tamil
        try:
            translator = Translator()
            tamil_text = translator.translate(hindi_text, src="hi", dest="ta").text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

        # Step 3: Tamil text-to-speech (gTTS only supports MP3 natively)
        mp3_file = temp_path.replace(".wav", ".mp3")
        try:
            tts = gTTS(text=tamil_text, lang="ta")
            tts.save(mp3_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

        # Step 4: Convert MP3 → requested format (if needed)
        out_file = mp3_file
        if format != "mp3":
            sound = AudioSegment.from_mp3(mp3_file)
            out_file = mp3_file.replace(".mp3", f".{format}")
            sound.export(out_file, format=format)

        # Return Tamil audio + text
        return JSONResponse({
            "hindi_text": hindi_text,
            "tamil_text": tamil_text,
            "download_url": f"/download_audio?path={out_file}"
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/download_audio")
def download_audio(path: str):
    """Serve generated Tamil audio file"""
    if os.path.exists(path):
        ext = path.split(".")[-1]
        return FileResponse(path, media_type=f"audio/{ext}", filename=f"tamil_output.{ext}")
    return {"error": "Audio file not found."}
