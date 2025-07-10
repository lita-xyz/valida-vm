#!/bin/bash

set -e

if [ $# -ne 4 ]; then
    echo "Error: This script requires exactly 4 file paths as arguments."
    echo "Usage: $0 <ELF> <STDOUT> <VK> <PROOF>"
    exit 1
fi

# Function to write a 4-byte little-endian integer to a file
write_little_endian_int() {
    local value=$1
    local output_file=$2

    # Write the 4-byte little-endian representation
    printf "\\$(printf '%03o' $(( $value & 0xFF )))" >> "$output_file"
    printf "\\$(printf '%03o' $(( ($value >> 8) & 0xFF )))" >> "$output_file"
    printf "\\$(printf '%03o' $(( ($value >> 16) & 0xFF )))" >> "$output_file"
    printf "\\$(printf '%03o' $(( ($value >> 24) & 0xFF )))" >> "$output_file"
}

ELF_FILE="$1"
STDOUT_FILE="$2"
VK_FILE="$3"
PROOF_FILE="$4"
OUTPUT_FILE="recursive-verifier-input"

rm -f $OUTPUT_FILE

INPUT_FILE=$ELF_FILE

FILE_SIZE=$(stat -c %s "$INPUT_FILE" 2>/dev/null || stat -f %z "$INPUT_FILE")
write_little_endian_int "$FILE_SIZE" "$OUTPUT_FILE"
cat "$INPUT_FILE" >> "$OUTPUT_FILE"

INPUT_FILE=$STDOUT_FILE

FILE_SIZE=$(stat -c %s "$INPUT_FILE" 2>/dev/null || stat -f %z "$INPUT_FILE")
write_little_endian_int "$FILE_SIZE" "$OUTPUT_FILE"
cat "$INPUT_FILE" >> "$OUTPUT_FILE"

INPUT_FILE=$VK_FILE

FILE_SIZE=$(stat -c %s "$INPUT_FILE" 2>/dev/null || stat -f %z "$INPUT_FILE")
write_little_endian_int "$FILE_SIZE" "$OUTPUT_FILE"
cat "$INPUT_FILE" >> "$OUTPUT_FILE"

INPUT_FILE=$PROOF_FILE

FILE_SIZE=$(stat -c %s "$INPUT_FILE" 2>/dev/null || stat -f %z "$INPUT_FILE")
write_little_endian_int "$FILE_SIZE" "$OUTPUT_FILE"
cat "$INPUT_FILE" >> "$OUTPUT_FILE"

