#!/bin/bash

# Define the text file to be read
text_file="/path/to/optimizer/out1.txt"

# Define the output file where the matching lines will be stored
output_file="/path/to/optimizer/gp.txt"

# Use grep to extract lines containing any of the specified strings
# awk '
#   /metrics/ && /True/ {print; getline; print}
# ' "$text_file" > "$output_file"
# sed -i 's/True//g' "$output_file"
# sed -i "s/.*nA1/nA1/" "$output_file"
# sed -i "s/.*metrics/metrics/" "$output_file"

awk '
  /metrics/ && /True/ { 
    if (getline nextline && getline nextnextline && getline nextnextnextline) { 
      if (nextnextnextline ~ /reward/) {
        print $0; 
		print nextline;
        print nextnextline;
        print nextnextnextline;
      }
    }
  }
' "$text_file" > "$output_file"

sed -i 's/True//g' "$output_file"

