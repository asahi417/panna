# NERVOUS
This is the project nervous. 


## Setup
- Install PANNA
```shell
git clone git@github.com:asahi417/panna.git
cd panna
pip install -e .
```

- Move to project folder

```shell
cd projects/nervous
```

## Generation Video
The [generate_video.py](https://github.com/asahi417/panna/blob/main/projects/nervous/generate_video.py) is the main script.

```
usage: generate_video.py [-h] -v VIDEO [-f FPS] [-s START] -e END -o OUTPUT [--sd SD] --prompt1 PROMPT1 [--prompt2 PROMPT2] [--prompt3 PROMPT3] [--prompt4 PROMPT4]

I am NERVOUS!

options:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to the input video.
  -f FPS, --fps FPS     FPS of the output video.
  -s START, --start START
                        Start of the video.
  -e END, --end END     End of the video.
  -o OUTPUT, --output OUTPUT
                        Path to the output video.
  --sd SD               SD when merging prompt embeddings.
  --prompt1 PROMPT1     The 1st prompt.
  --prompt2 PROMPT2     The 2nd prompt.
  --prompt3 PROMPT3     The 3rd prompt.
  --prompt4 PROMPT4     The 4th prompt.
```

### Examples

- From Cubism to Impressionism
```shell
python generate_video.py \
    -v "sample.mp4" \
    -f 10 \
    -s 0 \
    -e 2 \
    -o "sample_1.mp4" \
    --sd 0.2 \
    --prompt1 "Inspiring, surrealism, expressive, cubism, HQ, 4K" \
    --prompt2 "Organic, nature, forest, meditative, Monet, impressionist"
```


## Refine Video
The [refine_video.py](https://github.com/asahi417/panna/blob/main/projects/nervous/refine_video.py) is a sub-script to refine the genrated video.

```
usage: refine_video.py [-h] -v VIDEO [-p PROMPT] -o OUTPUT

Level up your NERVOUS video!

options:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to the input video.
  -p PROMPT, --prompt PROMPT
                        Prompt.
  -o OUTPUT, --output OUTPUT
                        Path to the output video.
```


### Examples
- From Cubism to Impressionism
```shell
python refine_video.py \
    -v "sample_1.mp4" \
    -o "sample_1.refined.up.mp4" \
    -u
```
