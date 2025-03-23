from panna.util import get_frames, save_frames

if __name__ == '__main__':
    frames, fps, size = get_frames(
        "sample.mp4",
        start_sec=7,
        end_sec=13,
        fps=5,
        size=(512, 512),
        return_array=True
    )
    print(len(frames))
    save_frames("sample_cropped.mp4", frames, fps, size)
