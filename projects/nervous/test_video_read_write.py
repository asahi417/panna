from panna.util import get_frames, save_frames


samples = [
    "sample.mp4",
    "sample.mov"
]
if __name__ == '__main__':
    for sample in samples:
        frames, fps, size = get_frames(
            sample,
            start_sec=7.5,
            end_sec=13.5,
            fps=5,
            size=(512, 512),
            return_array=True
        )
        save_frames(f"{sample.replace('.', '_')}.cropped.mp4", frames, fps, size)
