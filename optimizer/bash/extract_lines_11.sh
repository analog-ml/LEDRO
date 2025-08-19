#!/bin/bash

# Define the text file to be read
text_file="/path/to/optimizer/out11.txt"

# Define the output file where the matching lines will be stored
output_file="/path/to/optimizer/gp11.txt"

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

grep "reward" /path/to/optimizer/gp11.txt | sed 's/reward //' | sort -nr | head -1 > /path/to/optimizer/top_reward11.txt


awk '/metrics/ || /nA/ || /MM/ || /reward/' "$output_file" > /path/to/optimizer/temp_extracted_lines11.txt
grep "reward" /path/to/optimizer/temp_extracted_lines11.txt | sed 's/reward //' | sort -nr | head -5 > /path/to/optimizer/top_5_rewards11.txt
declare -A rewards_map
while read -r reward; do
    rewards_map["$reward"]=1
done < /path/to/optimizer/top_5_rewards11.txt

# Initialize variables to store lines
metrics_line=""
na_line=""
gm_line=""
# Iterate through the extracted lines and print the required lines
declare -a data_blocks
while read -r line; do
    if [[ "$line" =~ metrics ]]; then
        metrics_line="$line"
    elif [[ "$line" =~ nA ]]; then
        na_line="$line"
    elif [[ "$line" =~ MM ]]; then
        gm_line="$line"
    elif [[ "$line" =~ reward ]]; then
        reward_value=$(echo "$line" | sed 's/reward //')
        if [[ ${rewards_map["$reward_value"]} ]]; then
            #echo "$metrics_line"
            #echo "$na_line"
            #echo "$gm_line"
            #echo "$line"
 	data_blocks+=("$reward_value $metrics_line\n$na_line\n$gm_line\n$line")
        fi
    fi
done < $output_file


IFS=$'\n' sorted_blocks=($(sort -r -n <<<"${data_blocks[*]}"))
{
    for block in "${sorted_blocks[@]}"; do
        echo -e "${block#* }"
    done
} > "/path/to/optimizer/top_5_rewards11.txt"
