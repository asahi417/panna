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

## Run
The [main.py](./projects/nervous/main.py) is the main script.

```
usage: main.py [-h] [-v VIDEO] [-f FPS] [-s START] [-e END] [-o OUTPUT] [--sd SD] [--prompt1 PROMPT1] [--prompt2 PROMPT2] [--prompt3 PROMPT3] [--prompt4 PROMPT4]

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
  --prompt4 PROMPT4     The 1st prompt.
```

### Examples
```shell
python main.py \
-v sample.mp4 \
-f 5 \
-s 0 \
-e 5 \
-o sample_exmaple_1.mp4 \
--sd 0.2 \
--prompt1 "flower blooming, reflection, HQ, super realistic, geometric, artistic, creative, inspiring" \
--prompt2 "flower, dark, depression, winter, aesthetic" \
--prompt3 "flower blooming, nature, peaceful, beautiful, summer, landscape"
```