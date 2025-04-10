# transcriber.py
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import os

def extract_audio(video_path, audio_path):
    print("🟡 Extracting audio...")
    video_clip = VideoFileClip(video_path)
    audio = video_clip.audio
    audio.write_audiofile(audio_path)
    video_clip.close()
    print("✅ Audio extracted and saved.")

def reduce_noise(audio_path):
    print("🟡 Reducing noise...")
    audio = AudioSegment.from_file(audio_path, format="mp3")
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    if not chunks:
        print("⚠️ No audio chunks found. Silence threshold might be too strict.")
    cleaned_audio = sum(chunks)
    cleaned_audio.export(audio_path, format="mp3")
    print("✅ Noise reduced and audio cleaned.")

def get_text_chunks(audio_path):
    print("🟡 Loading Whisper model...")
    model = whisper.load_model("base")
    print("✅ Model loaded. Transcribing...")
    result = model.transcribe(audio_path)
    segments = result['segments']

    text_chunks = []
    for segment in segments:
        text = segment['text'].strip()
        if text:
            text_chunks.append(text)

    print("✅ Transcription complete! Returning chunks.")
    return text_chunks

def transcribe_video_to_chunks(video_path):
    #video_path = "sampledvdo.mp4"  # 🔒 Fixed path (in same folder)
    audio_path = "extracted_audio.mp3"

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video not found at: {video_path}")

    extract_audio(video_path, audio_path)
    reduce_noise(audio_path)
    return get_text_chunks(audio_path)