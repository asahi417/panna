export PROMPT0="A portrait, painting, organic, nature, forest, meditative, Monet, impressionist, warm color, happiness cheerful, HQ, 4k"
export PROMPT1="Cubism, geometric, Picasso, 20th, modern art, warm color, happiness cheerful, HQ, 4k"
#export PROMPT1="Abstraction, distorted noisy picture, chaos, symbolism, sacred, dripping, pattern, HQ, 4k"
export FPS=15

python generate_video.py \
  --prompt-0 ${PROMPT0} \
  --prompt-1 ${PROMPT1} \
  -f ${FPS} \
  -v "source_input/vertical.mp4" \
  -a "audio/negative_ambient_9.wav" \
  -o "output/vertical_1.mp4" \
  --height 1080 \
  --width 1080 \
  --debug
ffmpeg \
  -i "output/vertical_1.mp4" \
  -i "audio/negative_ambient_9.wav" \
  -c:v copy \
  -vcodec libx264 \
  -shortest \
  -map 0:v:0 \
  -map 1:a:0 \
  -c:a aac "output/vertical_1.audio.mp4"
ffmpeg -i "output/vertical_1.audio.mp4" \
  -vf "crop=690:1080:195:0" \
  "output/vertical_1.audio.crop.mp4"
