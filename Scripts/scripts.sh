#!/bin/bash

img_domain='DCT RGB'
img_format='JPEG PNG'
target_list='bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'

for domain in $img_domain; do
    for format in $img_format; do
        # Check for the RGB-JPEG combination and skip it
        if [ "$domain" = "RGB" ] && [ "$format" = "JPEG" ]; then
            continue
        fi

        for target in $target_list; do
            accelerate launch main.py --domain $domain --img_format $format --target $target
        done
    done
done