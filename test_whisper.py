from app.services import audio,subtitle,voice

# mac
origin_video_path = "/Users/monkeygeek/Downloads/whisper_test1.webm"
output_audio_path = "/Users/monkeygeek/Downloads/whisper_test1.wav"
subtitle_file_path = "/Users/monkeygeek/Downloads/whisper_test1.srt"
audio_temp_path = "/Users/monkeygeek/Downloads/tmp"
voice_output_path = "/Users/monkeygeek/Downloads"


if __name__ == '__main__':
    # split audio from video
    #audio.get_audio_from_video(origin_video_path,output_audio_path)

    # whisper audio recognition
    #subtitle.create(output_audio_path, subtitle_file_path)

    # subtitle to voice
    subtitle_texts = subtitle.file_to_subtitles(subtitle_file_path)
    voice.subtitle_to_voice(subtitles=subtitle_texts,
                            temp_path=audio_temp_path,
                            voice_name="zh-CN-YunxiNeural",
                            voice_rate=float(1.0),
                            voice_pitch=float(1.0),
                            voice_volume=float(1.0),
                            out_path=voice_output_path
                            )
