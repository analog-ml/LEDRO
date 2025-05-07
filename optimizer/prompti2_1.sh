#!/bin/bash

# Input and output file paths
input_file="/path/to/optimizer/gp.txt"
output_file="/path/to/optimizer/outputgp2.txt"
output_file1="/path/to/optimizer/outputgp3.txt"
target_file="/path/to/LLM/sample-qa-data_l.json"
history_file="/path/to/optimizer/top_5_history.txt"
json_output=""

while IFS= read -r line1 && IFS= read -r line2 && IFS= read -r line3 && IFS= read -r line4; do
    json_output+="$line1 with $line2 and transistor regions $line3 and $line4. Then, " #with circuit parameters $line3
done < "$input_file"
json_output=${json_output%. Then, }
echo -e "$json_output" > "$output_file"
replacement_content=$(<"$output_file")
escaped_replacement_content=$(echo "$replacement_content" | sed 's/[&/\]/\\&/g')

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

# Clean up temporary files
rm /path/to/optimizer/temp_extracted_lines.txt #top_5_rewards.txt

json_output1=""

while IFS= read -r line1 && IFS= read -r line2 && IFS= read -r line3 && IFS= read -r line4; do
    json_output1+="$line1 with $line2 and transistor regions $line3 and $line4. Then, " #with circuit parameters $line3
done < "/path/to/optimizer/top_5_rewards1.txt"
json_output1=${json_output1%. Then, }
echo -e "$json_output1" > "$output_file1"
replacement_content1=$(<"$output_file1")
escaped_replacement_content1=$(echo "$replacement_content1" | sed 's/[&/\]/\\&/g')

line_count=$(wc -l < "$input_file")

if [ "$line_count" -eq 0 ]; then
  replace_number="0"
  replacement=""
  cp /path/to/LLM/sample-qa-data_l_basic_1.json /path/to/LLM/sample-qa-data_l.json
  sed -i "s/REPLACENUMBER/$replace_number/g" "$target_file"
  sed -i "s/REPLACEMENT/$replacement/g" "$target_file"
elif [ "$line_count" -lt 20 ]; then
  replace_number="$((line_count / 4))"
  replacement="$escaped_replacement_content"
  cp /path/to/LLM/sample-qa-data_l_basic_1.json /path/to/LLM/sample-qa-data_l.json
  sed -i "s/REPLACENUMBER/only $replace_number/g" "$target_file"
  sed -i "s/REPLACEMENT/These are the points with their reward: $replacement/g" "$target_file"
else
  replacement="$escaped_replacement_content1"
  cp /path/to/LLM/sample-qa-data_l_basic_2.json /path/to/LLM/sample-qa-data_l.json
  sed -i "s/REPLACEMENT/$replacement/g" "$target_file"
fi


line_count1=$(wc -l < "$history_file")
if [ "$line_count1" -eq 0 ]; then
  replacement=""
  sed -i "s/REWARDSTATEMENT. /$replacement/g" "$target_file"
else
  max_value=$(sort -n $history_file | tail -n 1)
  replacement="As a reminder, the previous range you'd sent me before this one had given a best reward of"
  sed -i "s/REWARDSTATEMENT/$replacement $max_value. So, please consider that as well before giving your answer/g" "$target_file"
fi

cat /path/to/optimizer/top_5_rewards1.txt /path/to/optimizer/top_reward1.txt > /path/to/optimizer/combined_temp.txt
min_reward=""
line1=""
line2=""
line3=""
min_block=""
while IFS= read -r line; do
    if [[ $line == metrics* ]]; then
        prev1="$line"
    elif [[ $line == nA* ]]; then
        prev2="$line"
    elif [[ $line == MM* ]]; then
        prev3="$line"
    elif [[ $line == reward* ]]; then
        reward_value=$(echo $line | awk '{print $2}')
        
        if [[ -z $min_reward ]] || (( $(echo "$reward_value > $min_reward" | bc -l) )); then
            min_reward=$reward_value
            min_block="$prev1"$'\n'"$prev2"$'\n'"$prev3"$'\n'"$line"
        fi
    fi
done < "/path/to/optimizer/combined_temp.txt"

# Save the block with the minimum reward to the output file
echo "$min_block" > "/path/to/optimizer/top_reward1.txt"


json_output2=""

while IFS= read -r line1 && IFS= read -r line2 && IFS= read -r line3 && IFS= read -r line4; do
    json_output2+="$line1 with $line2 and transistor regions $line3 and $line4" #with circuit parameters $line3 
done < "/path/to/optimizer/top_reward1.txt"
json_output3=$(echo "$json_output2" | sed 's/[&/\]/\\&/g')
sed -i "s/TOPREWARD/$json_output3/g" "$target_file"


grep "reward " "/path/to/optimizer/top_reward1.txt" | awk '{print $2}' > "/path/to/optimizer/top_reward.txt"
