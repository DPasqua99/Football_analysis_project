from utils import save_video, read_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames)

    # save video
    save_video(video_frames, 'output_videos/output_video.avi')

if __name__=='__main__':
    main()