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


