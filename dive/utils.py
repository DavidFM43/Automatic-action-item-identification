import os
import subprocess

import pytube as pt


def download_yt_video(url, output_dir="."):
    """Downloads the audio pitch from a YouTube video and the convert it to .wav format"""

    yt = pt.YouTube(url)
    video_title = yt.title
    output_path = os.path.join(output_dir, f"{video_title}.mp3")
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(filename=output_path)

    wav_file_path = convert_to_wav(output_path)
    # remove old file
    os.remove(output_path)

    return wav_file_path


def convert_to_wav(audio_path: str):
    """Convert an audio file to .wav format"""

    _, file_ending = os.path.splitext(f"{audio_path}")
    wav_file_path = audio_path.replace(file_ending, ".wav")
    subprocess.run(
        f'ffmpeg -i "{audio_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{wav_file_path}"',
        shell=True,
        # supress any output
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_file_path


def convert_time(seconds):
    """Convert time in seconds to HH:MM:SS format."""

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{str(hours).rjust(2, '0')}:{str(minutes).rjust(2, '0')}:{str(seconds).rjust(2, '0')}"
