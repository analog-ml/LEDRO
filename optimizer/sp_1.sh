input_file="/path/to/LLM/qa_output_amp_l.txt"
output_file="/path/to/optimizer/qa_output_amp_l.txt"
output_file1="/path/to/optimizer/qa_output_amp_l1.txt"

# Extract lines starting with "nA", "nB", or "vbias"
grep -E '^(nA|nB|vbias|cc)' "$input_file" > "$output_file"

python /path/to/optimizer/transform_file.py $output_file $output_file1

temp_file=$output_file
awk '{print "  " $0}' "$output_file1" > "$temp_file"

file1="$temp_file"
file2="/path/to/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode_basic.yaml"
file3="/path/to/optimizer/working_current/spectre_simulator/spectre/specs_list_read/fully_differential_folded_cascode.yaml"

temp_file1=$(mktemp)
found=0
while IFS= read -r line; do
  echo "$line" >> "$temp_file1"
  if [[ "$line" == params:* ]]; then
    found=1
    # After finding "params:", copy contents of file1
    cat "$file1" >> "$temp_file1"
  fi
done < "$file2"

if [ "$found" -eq 0 ]; then
  cat "$file1" >> "$temp_file1"
fi

mv "$temp_file1" "$file3"

file1="/path/to/LLM/qa_output_amp_l.txt"
file2="/path/to/optimizer/working_current/sample/random_sample_turbo_1.py"
temp_file="/path/to/LLM/tempfile.txt"

declare -A range_values

# Read ranges from file1 and store them in the associative array
while IFS=': ' read -r param range; do
    param=$(echo $param | tr -d ' ,')
    range_values[$param]=$(echo $range | tr -d '[]')
done < <(grep -oP '^[^:]+: \[[^]]+\]' "$file1")

# Read file2 line by line and replace the ranges
while IFS= read -r line; do
    for param in "${!range_values[@]}"; do
        pattern="${param}_range = \([^)]*\)"
        if [[ $line =~ $pattern ]]; then
            # Replace the range in the line
            new_range=${range_values[$param]}
            line=$(echo "$line" | sed "s/${pattern}/${param}_range = ($new_range/")
        fi
    done
    echo "$line" >> "$temp_file"
done < "$file2"

# Replace the original file2 with the updated content
mv "$temp_file" "$file2"


