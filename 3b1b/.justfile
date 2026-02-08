@build:
  # uvx manim render string_art_explainer.py -qm --format=mp4 -a
  ls -1 media/videos/string_art_explainer/1080p30/Scene*.mp4 | sort -V | sed "s/^/file '/" | sed "s/$/'/" > concat.txt && ffmpeg -y -f concat -safe 0 -i concat.txt -c copy string_art_explainer.mp4 && rm concat.txt

@clean:
  rm -rf media
