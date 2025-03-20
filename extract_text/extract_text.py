import moviepy as mp
import whisper
import os

class AudioTranscriber:
    
    def __init__(self, model_size="small"):
        """
        Initializes the Whisper model.
        :param model_size: The size of the Whisper model (options: "tiny", "base", "small", "medium", "large").
        """
        self.model = whisper.load_model(model_size)
        # self.__output_audio
        # self.__text = ""


    def extract_audio(self, video_path, output_audio_path="./resources/output_audio.wav"):
        """
        Extracts audio from the given video file.
        :param video_path: Path to the input video file.
        :param output_audio_path: Path to save the extracted audio file.
        :return: Path of the extracted audio file.
        """
        try:
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(output_audio_path, codec="pcm_s16le")
            return output_audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """
        Transcribes the extracted audio using Whisper.
        :param audio_path: Path to the extracted audio file.
        :return: Transcribed text.
        """
        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

