import re

def process_line(line):
    if line.startswith('nA') or line.startswith('vbias'):
        # Add !!python/tuple and !!float
        parts1 = line.split(':', 1)
        parts = re.findall(r'[\d.]+(?:e-?\d+)?', parts1[1])
        numbers = [f'!!float {part}' for part in parts]
        return f"{line.split(':')[0]}: !!python/tuple [{', '.join(numbers)}, 13]\n"
    elif line.startswith('nB'):
        # Add !!python/tuple and difference calculation
        parts = re.findall(r'[\d.]+', line)
        num1 = int(parts[1])
        num2 = int(parts[2])
        diff = num2 - num1 + 1
        return f"{line.split(':')[0]}: !!python/tuple [{num1}, {num2}, {diff}]\n"
    else:
        return line

def transform_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            outfile.write(process_line(line))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python transform_file.py input.txt output.txt")
    else:
        transform_file(sys.argv[1], sys.argv[2])

