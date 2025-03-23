from panna.util import get_frames, save_frames

sample = "/Users/asahi/Downloads/763574630.249415.mp4"
if __name__ == '__main__':
    frames, fps, size = get_frames(
        sample,
        start_sec=7,
        end_sec=13,
        fps=5,
        size=(512, 512),
        return_array=True
    )
    print(len(frames))
    save_frames("sample_cropped.mp4", frames, fps, size)
