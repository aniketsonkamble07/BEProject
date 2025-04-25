import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import functools

# Cache the Whisper model to avoid reloading
@functools.lru_cache(maxsize=1)
def load_whisper_model():
    print("🟡 Loading Whisper model...")
    model = whisper.load_model("base")
    print("✅ Model loaded.")
    return model

def extract_audio(video_path, audio_path):
    """Extract audio from video with caching"""
    if os.path.exists(audio_path):
        print("ℹ️ Using cached audio file")
        return
        
    print("🟡 Extracting audio...")
    with VideoFileClip(video_path) as video_clip:
        audio = video_clip.audio
        audio.write_audiofile(audio_path)
    print("✅ Audio extracted and saved.")

def reduce_noise(audio_path, overwrite=False):
    """Clean audio with intelligent caching"""
    cleaned_path = audio_path.replace(".mp3", "_cleaned.mp3")
    
    if os.path.exists(cleaned_path) and not overwrite:
        print("ℹ️ Using cached cleaned audio")
        return cleaned_path
        
    print("🟡 Reducing noise...")
    audio = AudioSegment.from_file(audio_path, format="mp3")
    chunks = split_on_silence(
        audio, 
        min_silence_len=500, 
        silence_thresh=-40,
        keep_silence=200  # Keep small gaps between words
    )
    
    if not chunks:
        print("⚠️ No audio chunks found. Returning original audio.")
        return audio_path
        
    cleaned_audio = sum(chunks)
    cleaned_audio.export(cleaned_path, format="mp3")
    print("✅ Noise reduced and audio cleaned.")
    return cleaned_path

def transcribe_audio(audio_path):
    """Transcribe audio with optimized settings"""
    model = load_whisper_model()
    print("🟡 Transcribing audio...")
    
    # Use faster decoding with quality tradeoffs
    result = model.transcribe(
        audio_path,
        fp16=False,  # Disable if not using GPU
        language='en',
        initial_prompt="English transcription",  # Improves accuracy
        word_timestamps=False  # Faster without word-level timings
    )
    
    # Filter and clean text chunks
    text_chunks = [
        segment['text'].strip() 
        for segment in result['segments'] 
        if segment['text'].strip()
    ]
    
    print(f"✅ Transcription complete! Got {len(text_chunks)} chunks.")
    return text_chunks

def transcribe_video_to_chunks(video_path):
    """Main processing pipeline with caching"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video not found at: {video_path}")
    
    # Use consistent temp file naming
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = f"temp_{base_name}.mp3"
    
    # Processing pipeline
    extract_audio(video_path, audio_path)
    cleaned_audio = reduce_noise(audio_path)
    
    try:
        return transcribe_audio(cleaned_audio)
    finally:
        # Cleanup temp files (comment out for debugging)
        for f in [audio_path, cleaned_audio]:
            if f != cleaned_audio and os.path.exists(f):
                os.remove(f)