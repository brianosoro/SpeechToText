import os
import whisper
import soundfile as sf
import noisereduce as nr
from pathlib import Path
from pydub import AudioSegment, effects
from tqdm import tqdm


MODEL_SIZE = "medium"
CHUNK_LENGTH_MS = 30 * 1000
OVERLAP_MS = 3000
temp_files = []


def convert(input_path: Path) -> Path:

    print(f"Converting: {input_path.name}")

    output_path = input_path.with_suffix('.wav')

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = effects.normalize(audio)
        audio.export(output_path, format="wav")

        temp_files.append(output_path)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to process audio file: {e}")


def reduce_noise(input_wav: Path) -> Path:

    print("Noise reduction...")

    cleaned_path = input_wav.with_name(f"{input_wav.stem}_cleaned.wav")
    data, rate = sf.read(input_wav)
    reduced_noise_data = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.85)
    sf.write(cleaned_path, reduced_noise_data, rate)
    temp_files.append(cleaned_path)
    return cleaned_path


def create_chunks(audio_path: Path):
    audio = AudioSegment.from_wav(audio_path)
    total_length = len(audio)

    print(f"Splitting audio into {CHUNK_LENGTH_MS / 1000}s chunks....")

    for start in range(0, total_length, CHUNK_LENGTH_MS - OVERLAP_MS):
        end = min(start + CHUNK_LENGTH_MS, total_length)
        chunk = audio[start:end]
        chunk_name = audio_path.with_name(f"temp_chunk_{start}_{end}.wav")
        chunk.export(chunk_name, format="wav")
        temp_files.append(chunk_name)

        yield chunk_name


def transcribe_audio(cleaned_path: Path, model_name: str) -> str:

    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    transcripts = []
    chunks = list(create_chunks(cleaned_path))

    print("Start transcripting...")
    for chunk_file in tqdm(chunks, desc="Processing chunks", unit="segment"):
        result = model.transcribe(
            str(chunk_file),
            fp16=False,  # If you have a  GPU, set this to TRUE
            language=None
        )
        text = result.get("text", "").strip()
        transcripts.append(text)

    return "\n".join(transcripts)


def cleanup():
    print("Delete temporary files...")
    for file_path in temp_files:
        if file_path.exists():
            try:
                os.remove(file_path)
            except OSError:
                print(f"Warning: Could not delete {file_path}")



if __name__ == "__main__":

    INPUT_FILE = Path("audio_four.m4a")

    try:
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Could not find file: {INPUT_FILE}")

        wav_file = convert(INPUT_FILE)
        clean_file = reduce_noise(wav_file)
        full_text = transcribe_audio(clean_file, MODEL_SIZE)

        print("\n" + "=" * 40)
        print("FINAL TRANSCRIPTION")
        print("=" * 40)
        print(full_text[:500] + "... (truncated for display)")  # Print preview

        output_filename = INPUT_FILE.with_name(f"{INPUT_FILE.stem}_transcript.txt")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Save transcription to: {output_filename}")

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # This runs whether the script succeeds OR fails
        cleanup()