import whisper
import glob
import os

class WhisperTranscriber:
    def __init__(self, model_size="small"):
        self.model = whisper.load_model(model_size)

    def find_latest_wav(self, folder):
        wav_files = glob.glob(os.path.join(folder, '*.wav'))
        if not wav_files:
            raise FileNotFoundError(f"No wav files found in {folder}.")
        latest_file = max(wav_files, key=os.path.getctime) #could return the latest established .wav file
        return latest_file

    def transcribe(self, audio_path, word_timestamps=True):
        result = self.model.transcribe(audio_path, word_timestamps=word_timestamps)
        return result

    def word_timestamps(self, result):
        output_with_word_timestamps = []
        print(result["text"])
        for segment in result['segments']:
            for word in segment['words']:
                output = f"{word['word']} [{word['start']:.2f}s - {word['end']:.2f}s]"
                print(output)
                output_with_word_timestamps.append(output)
        return output_with_word_timestamps

if __name__ == "__main__":
    transcriber = WhisperTranscriber(model_size="base")
    audio_path = transcriber.find_latest_wav(folder="sound_storage")
    result = transcriber.transcribe(audio_path)
    transcriber.print_word_timestamps(result)