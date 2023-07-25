from pathlib import Path
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
with open(Path.cwd() / "src" / "lost_debit_card.wav", "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)

"""to enable your own recording on a non codespaces
freq = 44100
duration = 5
import sounddevice as sd
from scipy.io.wavfile import write
recording = sd.rec(int(duration * freq),
                   samplerate=freq, channels=2)
 
sd.wait()
write("my_audio.wav", freq, recording)
"""
