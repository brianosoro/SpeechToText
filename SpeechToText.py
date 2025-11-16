import whisper
from pydub import AudioSegment, effects
import noisereduce as nr
import soundfile as sf
import os
from tqdm import tqdm

#Input & output setup
input_file = "audio_four.m4a" #set your own audio file
base_name, _ = os.path.splitext(input_file)
wav_file = f"{base_name}.wav"
cleaned_file = f"{base_name}_cleaned.wav"

#Convert M4A to WAV & normalize
print("Converting and normalizing audio")
audio = AudioSegment.from_file(input_file)
audio = audio.set_channels(1)
audio = effects.normalize(audio)
audio.export(wav_file, format="wav")

#Noise reduction
print("Reducing background noise")
data, rate = sf.read(wav_file)
reduced = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.85)
sf.write(cleaned_file, reduced, rate)

#Load Whisper model
print("Loading Whisper model")
model = whisper.load_model("large")  #tiny, base, small, medium, large

#Split audio into manageable chunks ---
print("Splitting audio into chunks")
audio = AudioSegment.from_wav(cleaned_file)
chunk_length_ms = 30 * 1000  # 30 seconds per chunk
overlap_ms = 3000            # 3 second overlap between chunks

chunks = []
for start in range(0, len(audio), chunk_length_ms - overlap_ms):
    end = min(start + chunk_length_ms, len(audio))
    chunk = audio[start:end]
    chunk_name = f"{base_name}_chunk_{start//1000}-{end//1000}.wav"
    chunk.export(chunk_name, format="wav")
    chunks.append(chunk_name)

print(f"Created {len(chunks)} chunks.")

#Transcribe each chunk and merge the results
print("Transcribing chunks")
full_transcript = []

for i, chunk_file in enumerate(tqdm(chunks, desc="Transcribing", unit="chunk")):
    result = model.transcribe(
        chunk_file,
        fp16=False,
        temperature=0.0,
        beam_size=5,
        best_of=5,
        language=None  #automatically pick the language
    )
    text = result.get("text", "").strip()
    full_transcript.append(text)

#Merge and clean up
final_text = "\n".join(full_transcript)

print("\nFinal Transcription:")
print("=" * 60)
print(final_text)
print("=" * 60)

#Save transcription to file
output_file = f"{base_name}_transcription.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"\nSaved transcription to {output_file}")

#Clean the chunks
for chunk_file in chunks:
    os.remove(chunk_file)
print("Cleaned up temporary chunk files.")