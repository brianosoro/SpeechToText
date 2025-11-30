# SpeechToText

A Python snippet you can use to **transcribe audio files** on your **local machine** using OpenAI's Whisper model and have the output in a text file.

---

## Features

- **Easy to follow and less verbose**
- **Noise reduction** 
- **Extensible, build more interesting things on top** 
- **Lightweight as it can work on your local computer**  

---

## Use Cases

- Transcribe audio files to their text equivalent.

---

## Installation & Use

Clone the repository:
```bash
git clone https://github.com/brianosoro/SpeechToText.git
cd SpeechToText
#Add an m4a audio file in the directory 
```
Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Install FFmpeg(it is required by pydub):
```bash
brew install ffmpeg //For macOS
sudo apt install ffmpeg  //For Ubuntu
https://ffmpeg.org/download.html //For Windows
```
Start to transcribe, make sure you add an m4a audio file in the SpeechToText directory:
```bash
python SpeechToText.py
```
---

## Contributing

Contributions are welcome! Anyone should be able to initiate a pull request.

---

## License

MIT License - feel free to use in your projects!

---

## Contact

For questions or suggestions, please open an issue on GitHub.

---
