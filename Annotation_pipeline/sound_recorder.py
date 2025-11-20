import sounddevice as sd
import numpy as np
import time
import soundfile as sf
import datetime
import os


class SoundRecorder:

    def __init__(self, samplerate=16000, blocksize=16000, folder="sound_storage"):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.folder = folder
        self.audio_buffer = []
        self.stream = None
        self.is_recording = False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        # indata is a numpy array, which shape is (frames, channels), frames = blocksize
        # channels = 1
        self.audio_buffer.append(indata.copy()) #copy for subsequent processing

    def start_stream(self):
        self.audio_buffer = []
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def save(self):
        if not self.audio_buffer:
            print("No audio recorded!")
            return
        recorded_audio = np.concatenate(self.audio_buffer, axis=0).flatten() #concatenate audio data segments
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"output_{timestamp}.wav"
        os.makedirs(self.folder, exist_ok=True) #if dir doesn't exist, then makedir under the current path
        filepath = os.path.join(self.folder, filename)
        sf.write(filepath, recorded_audio, self.samplerate)
        print(f"Recording saved as: {filepath}")

    def record_audio(self):
        self.is_recording = True
        self.start_stream()
        try:
            while self.is_recording:
                time.sleep(0.5)
        finally:
            self.stop_stream()
            self.save()


# if __name__ == "__main__":
#     recorder = SoundRecorder()
#     recorder.record_audio()