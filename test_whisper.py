from app.services import audio,subtitle

# mac
origin_video_path = "/Users/monkeygeek/Downloads/whisper_test1.webm"
output_audio_path = "/Users/monkeygeek/Downloads/whisper_test1.wav"
subtitle_file_path = "/Users/monkeygeek/Downloads/whisper_test1.srt"


if __name__ == '__main__':
    # split audio from video
    #audio.get_audio_from_video(origin_video_path,output_audio_path)

    # whisper audio recognition
    subtitle.create(output_audio_path, subtitle_file_path)

