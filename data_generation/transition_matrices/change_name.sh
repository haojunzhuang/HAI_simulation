#!/bin/bash

# Loop through all files matching the pattern '*_day_sliding_window_size_*.pkl'
for file in *_day_sliding_window_size_*.pkl; do
    # Check if the file name actually contains the pattern
    if [[ $file =~ (.*)_day_sliding_window_size_3.pkl ]]; then
        # Construct the new file name by capturing groups and replacing the necessary part
        newname="${BASH_REMATCH[1]}_day_sw_3.pkl"
        # Rename the file
        mv "$file" "$newname"
        echo "Renamed $file to $newname"
    fi
done