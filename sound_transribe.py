import os
import platform
import subprocess
import sys
import gc
import torch
import whisper
from whisper.utils import get_writer

def check_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_os_specific_instructions():
    return {
        "Darwin": "\n\033[94mBrew install command:\nbrew install ffmpeg\033[0m",
        "Windows": "\n\033[94mChocolatey install command:\nchoco install ffmpeg\nScoop install command:\nscoop install ffmpeg\033[0m",
        "Linux": "\n\033[94mAPT install command:\nsudo apt install ffmpeg\033[0m"
    }.get(platform.system(), "Please install FFmpeg for your OS")

def optimize_system_settings():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(4)
    

def format_timestamp(seconds):
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{int(seconds // 3600):02}:{int(seconds % 3600 // 60):02}:{int(seconds % 60):02},{msecs:03}"

def validate_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        raise ValueError("Unsupported audio format")

def preprocess_audio(file_path):
    temp_file = "temp_audio.wav"
    cmd = [
        "ffmpeg",
        "-y", "-i", file_path,
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-hide_banner", "-loglevel", "error",
        temp_file
    ]
    subprocess.run(cmd, check=True)
    return temp_file

def main():
    try:
        print(f"\033[1mSystem: {platform.system()}\033[0m")
        
        if not check_ffmpeg_installed():
            print("\033[91mFFmpeg missing!\033[0m" + get_os_specific_instructions())
            return

        optimize_system_settings()
        
        print("\n\033[93mLoading Whisper Turbo...\033[0m")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("turbo", device=device)
        print(f"Using device: {device}")
        file_path = input("\n\033[1mAudio file path: \033[0m").strip()
        validate_file(file_path)
        
        print("\n\033[96mOptimizing audio...\033[0m")
        processed_file = preprocess_audio(file_path)

        print("\033[96mStarting transcription...\033[0m")
        result = model.transcribe(
            processed_file,
            fp16=torch.cuda.is_available(),
            language="pl",
            verbose=False,
            temperature=0,
            compression_ratio_threshold=2.4,
            best_of=1
        )

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        writer = get_writer("srt", ".")
        writer(result, f"{base_name}.srt")
        
        print(f"\n\033[1;92mSaved to:\033[0m {base_name}.srt")
        print(f"\033[95mProcessing time: {result['segments'][-1]['end']:.1f}s\033[0m")

    except Exception as e:
        print(f"\n\033[91mError: {e}\033[0m")
        sys.exit(1)
    finally:
        if 'processed_file' in locals():
            os.remove(processed_file)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
