import datetime
import pandas as pd
import random
import numpy as np
import tqdm
import os
import argparse
from entry_sampler import entry_sampler, toy_entry_sampler, quick_entry_sampler
from duration_sampler import duration_sampler, toy_duration_sampler, quick_duration_sampler
from path_sampler import path_sampler, toy_path_sampler


def generate_movement(transition_matrix_path, 
                      start_date_str, end_date_str, 
                      method, window_size=0, data_path=None, 
                      toy=False, num_toy_departments=10,
                      quick=False, num_entry_path=None, duration_path=None):
    if toy:
        my_entry_sampler = toy_entry_sampler()
        my_duration_sampler = toy_duration_sampler()
        my_path_sampler = toy_path_sampler(num_departments=num_toy_departments)
    elif quick:
        my_entry_sampler = quick_entry_sampler(num_entry_path)
        my_duration_sampler = quick_duration_sampler(duration_path)
        my_path_sampler = path_sampler(transition_matrix_folder_path=transition_matrix_path)
    else:
        my_entry_sampler = entry_sampler(data_path)
        my_duration_sampler = duration_sampler(data_path)
        my_path_sampler = path_sampler(data_path, transition_matrix_path)

    # Step 1: Determine Start and End Date
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

    daily_paths = {}

    # Step 2: For each day sample number of entry
    current_date = start_date
    while current_date <= end_date:
        if toy:
            num_entries = my_entry_sampler.sample()
        else:  
            num_entries = my_entry_sampler.sample(current_date.strftime('%Y-%m-%d'))

        one_day_paths = []
        for _ in range(num_entries):
            # Step 3: For each entry sample length of duration
            if toy:
                duration = my_duration_sampler.sample()
            else:
                duration = my_duration_sampler.sample()
            if duration == 0:
                duration = 1

            # Step 4: For each duration sample path
            if toy:
                path = my_path_sampler.sample(duration)
            else:
                path = my_path_sampler.sample(duration, method, window_size)

            one_day_paths.append(path)

        daily_paths[current_date.strftime('%Y-%m-%d')] = one_day_paths
        current_date += datetime.timedelta(days=1)

    return daily_paths

def path_to_movement(path_data):
    data_rows = []
    patient_id_counter = 1
    for start_date_str, paths in path_data.items():
        for path in paths:
            patient_id = f'A_{patient_id_counter}'
            patient_id_counter += 1
            for i in range(len(path) - 1):
                from_dept = path[i]
                to_dept = path[i + 1]
                step_date = pd.to_datetime(start_date_str) + pd.Timedelta(days=i)
                row = {'id': patient_id, 'date': step_date, 'from_department': from_dept, 'to_department': to_dept}
                data_rows.append(row)

    generated_movement = pd.DataFrame(data_rows)
    generated_movement.sort_values(by='date', inplace=True)
    generated_movement.reset_index(drop=True, inplace=True)

    return generated_movement

def run_generation(num_sample, transition_matrix_folder_path, output_folder_path, 
                   start_date_str, end_date_str, method, window_size=0, data_path=None, toy=False, num_toy_departments=10,
                   quick=False, num_entry_path=None, duration_path=None):
    for i in tqdm.tqdm(range(num_sample)):
        daily_paths = generate_movement(transition_matrix_folder_path, 
                                        start_date_str, end_date_str, 
                                        method, window_size, data_path, toy, num_toy_departments, 
                                        quick, num_entry_path, duration_path)
        generated_movement = path_to_movement(daily_paths)
        i = 1
        while True:
            if toy:
                file_path = os.path.join(output_folder_path, f"toy_{i}.pkl")
            elif method == "sliding_window":
                file_path = os.path.join(output_folder_path, f"sw_{window_size}_{i}.pkl")
            elif method == "shorter_only":
                file_path = os.path.join(output_folder_path, f"so_{i}.pkl")
            elif method == "longer_only":
                file_path = os.path.join(output_folder_path, f"lo_{i}.pkl")
            if not os.path.exists(file_path):
                generated_movement.to_pickle(file_path)
                print(f"DataFrame saved to {file_path}")
                break
            i += 1

if __name__ == "__main__":
    # NOTE: Change the Path here!
    transition_matrix_folder_path = "/Users/richardzhuang/Desktop/UCSF/HAI_simulation/movement_generation/deid_data/transition_matrices"
    output_folder_path = "/Users/richardzhuang/Desktop/UCSF/HAI_simulation/movement_generation/generated_movements"

    start_date_str = '2024-01-01'
    end_date_str = '2025-01-01'
    method = "sliding_window"
    window_size = 3

    data_path = "/Users/richardzhuang/Desktop/UCSF/HAI_simulation/simulation/data/movements_cleaned.csv"
    num_entry_path = "/Users/richardzhuang/Desktop/UCSF/HAI_simulation/movement_generation/deid_data/entries/num_entries.csv"
    duration_path = "/Users/richardzhuang/Desktop/UCSF/HAI_simulation/movement_generation/deid_data/durations/durations.csv"

    num_sample = 3

    # Alternative: Use Argparse (TBD)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--transition_matrix_folder_path", type=str)
    # args = parser.parse_args()

    # NOTE: If first time running, need to uncomment and run this block first to get pre-computed transition matrices
    # Possible methods are "longer_only", "shorter_only", and "sliding_window" with a window_size
    # my_path_sampler = path_sampler(data_path, transition_matrix_folder_path)
    # my_path_sampler.create_transition_matrices(method="sliding_window", window_size=3)
    # my_path_sampler.create_transition_matrices(method="longer_only")
    # my_path_sampler.create_transition_matrices(method="shorter_only")

    # If using existing entry and duration data, set quick=True, and don't need to set data path
    run_generation(num_sample, transition_matrix_folder_path, output_folder_path,
                start_date_str, end_date_str, method, window_size=window_size, 
                quick=True, num_entry_path=num_entry_path, duration_path=duration_path)

    # Else, give data path
    # run_generation(num_sample, transition_matrix_folder_path, output_folder_path,
    #                start_date_str, end_date_str, method, window_size=window_size, 
    #                data_path=data_path)