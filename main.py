from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import tempfile
import subprocess
import wave
import os
import logging

logger = logging.getLogger("app_logger")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

load_dotenv()
port = int(os.getenv("PORT", 8080))

app = FastAPI()

whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma', '.opus',
    '.amr', '.aiff', '.alac', '.pcm', '.webm', '.mp4', '.3gp', '.caf'
}

MAX_AUDIO_DURATION = 900

class TranscriptionPartResponse(BaseModel):
    part_id: int
    transcription: str
    language: str
    language_probability: Optional[float]
    subject: Optional[str]

def is_allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def convert_to_wav(input_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as input_tmp:
        input_tmp.write(input_bytes)
        input_tmp_path = input_tmp.name

    output_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_tmp_path = output_tmp.name
    output_tmp.close()

    cmd = [
        "ffmpeg", "-y", "-i", input_tmp_path,
        "-ar", "16000", "-ac", "1",
        "-f", "wav", output_tmp_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(input_tmp_path)

    if result.returncode != 0:
        os.remove(output_tmp_path)
        raise HTTPException(status_code=400, detail="Błąd konwersji pliku audio (ffmpeg).")

    return output_tmp_path


@app.post("/admin/audio-transcribe-part", response_model=TranscriptionPartResponse)
async def transcribe_audio_part(
        file: UploadFile = File(...),
        part_id: int = Form(...),
        subject: Optional[str] = Form(None),
        language: Optional[str] = Form(None)
):
    language = language or 'ru'
    subject = subject or 'Brak przedmiotu'
    filename = file.filename.lower()

    if not is_allowed_file(filename):
        raise HTTPException(status_code=400, detail="Nieobsługiwany format pliku audio.")

    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Plik audio jest zbyt duży. Maksymalnie 100 MB.")

    try:
        wav_path = convert_to_wav(audio_bytes)

        with wave.open(wav_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)

        if duration > MAX_AUDIO_DURATION:
            os.remove(wav_path)
            raise HTTPException(
                status_code=400,
                detail=f"Plik audio jest za długi: {duration:.2f} sekund. Maksymalnie 30 minut."
            )

        segments, info = whisper_model.transcribe(
            wav_path,
            beam_size=5,
            language=language,
            temperature=0.2
        )
        transcription = " ".join(segment.text.strip() for segment in segments).strip()
        os.remove(wav_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

    return TranscriptionPartResponse(
        part_id=int(part_id),
        transcription=str(transcription),
        language=str(info.language),
        language_probability=float(round(info.language_probability, 2)) if info.language_probability is not None else None,
        subject=str(subject) if subject else None
    )

# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=port,
#         reload=False,
#         timeout_keep_alive=900,
#         timeout_graceful_shutdown=900
#     )