# this is sparks implementation of compression, as long as the compress script is available in scripts folder
# juts run: 
# 1. chmod +x compress_splats_to_spz.sh
# 2. ./compress_splats_to_spz.sh

#!/bin/bash

# Directory containing .splat files
INPUT_DIR="/home/da10546y/NLF-GS/visualize/vis/avatars/splats_1_1"

# Iterate over all .splat files in the directory
for splat_file in "$INPUT_DIR"/*.splat; do
  if [ -f "$splat_file" ]; then
    # Run the compression command for each .splat file
    npm run assets:compress -- "$splat_file"

    echo "Compressed $splat_file to .spz"
  fi
done