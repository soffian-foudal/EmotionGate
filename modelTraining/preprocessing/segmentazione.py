from moviepy.editor import VideoFileClip
import os

def rimuovi_ripetizioni(video_folder):
    # Crea una cartella per i segmenti


    output_folder = os.path.join(video_folder, "videos_segmenti")
    os.makedirs(output_folder, exist_ok=True)

    # Elabora i video nella cartella di input
    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_folder, filename)
            print(f"Video path: {video_path}")

            # Carica il video
            clip = VideoFileClip(video_path)
            video_duration = clip.duration
            frame_ref = clip.get_frame(0)
            instants = [0]

            # Trova gli istanti con differenze significative tra i frame
            for t in range(1, int(video_duration)):
                frame = clip.get_frame(t)
                diff = frame - frame_ref
                sum_diff = diff.sum()

                if sum_diff > 100000:
                    instants.append(t)
                    frame_ref = frame

            # Aggiungi la durata del video come ultimo istante
            instants.append(video_duration)

            # Estrai il primo segmento e salvalo nella cartella di output
            start_time = instants[0]
            end_time = instants[1]
            segment_filename = f"segment_{filename}"
            segment_path = os.path.join(output_folder, segment_filename)
            segment_clip = clip.subclip(start_time, end_time)
            segment_clip.write_videofile(segment_path, codec="libx264")

    print("Elaborazione completata!")
    return output_folder

