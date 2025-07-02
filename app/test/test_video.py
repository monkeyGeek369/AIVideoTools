from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def split_video_every_seconds(input_video_path, output_folder,split_duration=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = VideoFileClip(input_video_path)
    duration = video.duration
    
    segments = int(duration // split_duration) + (1 if duration % split_duration != 0 else 0)
    
    for i in range(segments):
        start_time = i * split_duration
        end_time = min((i + 1) * split_duration, duration)

        subclip = video.subclip(start_time, end_time)
        
        output_path = os.path.join(output_folder, f"segment_{i+1}.mp4")
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Saved segment {i+1}: {output_path}")
    
    video.close()
    print("Video splitting completed!")


if __name__ == "__main__":
    split_video_every_seconds("/Users/monkeygeek/Downloads/test11.webm", "/Users/monkeygeek/Downloads",split_duration=1)




