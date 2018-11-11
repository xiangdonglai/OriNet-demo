ffmpeg \
  -i dslr_dance1/image.mp4 \
  -i dslr_dance1/side.mp4 \
  -filter_complex '[0:v]pad=iw*3:2*ih[int];[int][1:v]overlay=W/3:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  all.mp4
