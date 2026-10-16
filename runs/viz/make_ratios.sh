#!/bin/bash
# 4:5 master -> 16:9 (YouTube) and 9:16 (Reels/TikTok/Shorts): the frame centred over a blurred, darkened copy of itself
IN="$1"; BASE="${IN%_4x5.mp4}"
ffmpeg -nostdin -loglevel error -y -i "$IN" -filter_complex \
 "[0:v]split[a][b];[a]scale=1920:2400,crop=1920:1080,boxblur=28:3,eq=brightness=-0.22:saturation=0.8[bg];[b]scale=-2:1080[fg];[bg][fg]overlay=(W-w)/2:0,format=yuv420p[v]" \
 -map "[v]" -map 0:a? -c:v libx264 -crf 19 -preset medium -c:a copy -movflags +faststart "${BASE}_16x9.mp4"
ffmpeg -nostdin -loglevel error -y -i "$IN" -filter_complex \
 "[0:v]split[a][b];[a]scale=-2:1920,crop=1080:1920,boxblur=28:3,eq=brightness=-0.22:saturation=0.8[bg];[b]scale=1080:1350[fg];[bg][fg]overlay=0:(H-h)/2,format=yuv420p[v]" \
 -map "[v]" -map 0:a? -c:v libx264 -crf 19 -preset medium -c:a copy -movflags +faststart "${BASE}_9x16.mp4"
echo "wrote ${BASE}_16x9.mp4 ${BASE}_9x16.mp4"
