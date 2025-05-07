#!/bin/bash

# Input and output file paths
input_file="/path/to/optimizer/gp.txt"
output_file="/path/to/optimizer/outputgp.txt"
output_file1="/path/to/optimizer/outputgp3.txt"
target_file="/path/to/LLM/sample-qa-data_i.json"
history_file="/path/to/optimizer/top_5_history.txt"
json_output=""

awk '/metrics/ || /nA/ || /MM/ || /reward/' "$input_file" > /path/to/optimizer/temp_extracted_lines.txt
grep "reward" /path/to/optimizer/temp_extracted_lines.txt | sed 's/reward //' | sort -nr | head -5 > /path/to/optimizer/top_5_rewards.txt
declare -A rewards_map
while read -r reward; do
    rewards_map["$reward"]=1
done < /path/to/optimizer/top_5_rewards.txt

# rm combined_temp.txt

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
done < /path/to/optimizer/temp_extracted_lines.txt 
IFS=$'\n' sorted_blocks=($(sort -r -n <<<"${data_blocks[*]}"))
{
    for block in "${sorted_blocks[@]}"; do
        echo -e "${block#* }"
    done
} > "/path/to/optimizer/top_5_rewards1.txt"

tempfile=$(mktemp)
head -n 20 "/path/to/optimizer/top_5_rewards1.txt" > "$tempfile"
mv "$tempfile" "/path/to/optimizer/top_5_rewards1.txt"


grep "reward" /path/to/optimizer/temp_extracted_lines.txt | sed 's/reward //' | sort -nr | head -1 > /path/to/optimizer/top_reward.txt
declare -A rewards_map1
while read -r reward; do
    rewards_map1["$reward"]=1
done < /path/to/optimizer/top_reward.txt

metrics_line=""
na_line=""
gm_line=""
# Iterate through the extracted lines and print the required lines
while read -r line; do
    if [[ "$line" =~ metrics ]]; then
        metrics_line="$line"
    elif [[ "$line" =~ nA ]]; then
        na_line="$line"
    elif [[ "$line" =~ MM ]]; then
        gm_line="$line"
    elif [[ "$line" =~ reward ]]; then
        reward_value=$(echo "$line" | sed 's/reward //')
        if [[ ${rewards_map1["$reward_value"]} ]]; then
            echo "$metrics_line"
            echo "$na_line"
            echo "$gm_line"
            echo "$line"
	    break
        fi
    fi
done < /path/to/optimizer/temp_extracted_lines.txt > "/path/to/optimizer/top_reward1.txt"


# Clean up temporary files
rm /path/to/optimizer/temp_extracted_lines.txt #top_5_rewards.txt

json_output1=""

while IFS= read -r line1 && IFS= read -r line2 && IFS= read -r line3 && IFS= read -r line4; do #
    json_output1+="$line1 with $line2 and transistor regions $line3 and $line4. Then, " #with circuit parameters $line3
done < "/path/to/optimizer/top_5_rewards1.txt"
json_output1=${json_output1%. Then, }
echo -e "$json_output1" > "$output_file1"
replacement_content1=$(<"$output_file1")
escaped_replacement_content1=$(echo "$replacement_content1" | sed 's/[&/\]/\\&/g')


cat /path/to/LLM/sample-qa-data_i_basic.json > /path/to/LLM/sample-qa-data_i.json
search_string="REPLACEMENT"
sed -i "s|$search_string|$escaped_replacement_content1|g" "$target_file"
