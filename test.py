from extract_text.extract_text import AudioTranscriber

try:
    print("Step 1: Initializing transcriber")
    transcriber = AudioTranscriber(model_size='small')
    
    print("Step 2: Extracting audio from video")
    video_path = r"E:\BEProject\\resources\video.mp4"
    output_audio_path = r"E:\BEProject\resources\output_audio.wav"
    
    output_audio_path = transcriber.extract_audio(video_path=video_path, output_audio_path=output_audio_path)
    
    print("Step 3: Audio extraction complete")
    
    print("Step 4: Transcribing audio")
    text = transcriber.transcribe_audio(output_audio_path)
    
    print("Step 5: Transcription complete")
    print(text)

except Exception as e:
    print("An error occurred:", str(e))
