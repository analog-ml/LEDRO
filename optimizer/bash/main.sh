#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/path/to/optimizer/working_current"
export BASE_TMP_DIR=/path/to/optimizer/base_tmp/

mkdir /path/to/optimizer/outputfiles
output_base="output_file"
iterations=1
find "/path/to/optimizer/outputfiles" -type f -delete
for ((j=1; j<=iterations; j++))
do
	output_file="/path/to/optimizer/outputfiles/${output_base}_${j}.txt"
	cat /path/to/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode_initial.yaml > /path/to/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml
	rm /path/to/optimizer/out1.txt
	rm /path/to/optimizer/out11.txt
	rm /path/to/optimizer/out12.txt

	rm /path/to/optimizer/top_5_history.txt
	touch /path/to/optimizer/top_5_history.txt
	python /path/to/optimizer/working_current/sample/random_sample_turbo.py > /path/to/optimizer/dummy.txt
	cp /path/to/optimizer/out1.txt /path/to/optimizer/out1_i.txt


	bash /path/to/optimizer/extract_lines_1.sh
	bash /path/to/optimizer/prompti_1.sh

if [ ! -s "/path/to/optimizer/top_reward1.txt" ]; then
    for i in {1..4}; do
        echo "" >> "$output_file"
    done
else
    cat /path/to/optimizer/top_reward1.txt >> $output_file
fi

file_content=$(<"/path/to/optimizer/top_reward.txt")
if [ "$file_content" == "0" ]; then
  echo "0" > "/path/to/optimizer/iteration.txt"
  exit 0
fi

	python /path/to/LLM/llm_qa_csv_ex.py
	bash /path/to/optimizer/sp_1.sh

for (( i=1; i<=10; i++ ))
	do

	source /homes/username/.bashrc
	cat /path/to/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode_initial.yaml > /path/to/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml
	rm /path/to/optimizer/out1.txt

	source /homes/username/.bashrc
	python /path/to/optimizer/working_current/sample/random_sample_turbo_1.py > /path/to/optimizer/dummy1.txt

	bash /path/to/optimizer/extract_lines_1.sh
	bash /path/to/optimizer/prompti2_1.sh

if [ ! -s "/path/to/optimizer/top_reward1.txt" ]; then
    for i in {1..4}; do
        echo "" >> "$output_file"
    done
else
    cat /path/to/optimizer/top_reward1.txt >> $output_file
fi

file_content=$(<"/path/to/optimizer/top_reward.txt")
if [ "$file_content" == "0" ]; then
  echo "$i" > "/path/to/optimizer/iteration.txt"
  break
fi
if [[ $i -lt 10 ]]; then
	python /path/to/LLM/llm_qa_csv_ex1.py
	python /path/to/LLM/llm_qa_csv_ex2.py
fi
	bash /path/to/optimizer/sp_1.sh
	
	
	done

done
bash /path/to/optimizer/extract_lines_11.sh
bash /path/to/optimizer/extract_lines_12.sh

file1=$output_file
file2="/path/to/optimizer/top_reward11.txt"
file3="/path/to/optimizer/top_reward12.txt"
new_file="/path/to/optimizer.txt"


line4_file1=$(sed -n '4p' "$file1")
line44_file1=$(sed -n '44p' "$file1")
all_lines_file2=$(cat "$file2")
all_lines_file3=$(cat "$file3")
string2="reward "
line4_file11=$(echo "$line4_file1" | sed "s/$string2//g")
line44_file11=$(echo "$line44_file1" | sed "s/$string2//g")


{
  echo "$line4_file11 $all_lines_file2 $all_lines_file3 $line44_file11"
} > "$new_file"

rsync -a --delete /path/to/open-source/empty_dir/ /path/to/optimizer/base_tmp/ &

