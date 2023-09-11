#!/bin/bash

target_list='bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'

for target in $target_list; do
    accelerate launch main.py --target $target
done
