from matplotlib.colors import ListedColormap
from collections import Counter
from ..department import Department
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_prevalence(department: Department):
    """
    Plot the total number of patients and prevalence over time for a specific department,
    using index as a proxy for time.

    Parameters:
    - department_records: dict, a dictionary containing records for a department with keys 'total' and 'prevalence'.
    - department_name: str, the name of the department.
    """
    # Assuming the length of 'total' and 'prevalence' are the same

    department_records = department.records
    department_name = department.name

    time = list(range(len(department_records['total'])))
    
    df = pd.DataFrame({
        'Time': time,
        'Total': department_records['total'],
        'Prevalence': department_records['prevalence']
    })

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='value', hue='variable', data=pd.melt(df, ['Time']))
    plt.title(f'Total and Prevalence vs. Time for {department_name}')
    plt.ylabel('Count')
    plt.xlabel('Time (Unit)')
    plt.legend(title='Type')
    plt.show()


def plot_incidence(department: Department):
    """
    Plot the total number of patients and incidence over time for a specific department,
    using index as a proxy for time.

    Parameters:
    - department_records: dict, a dictionary containing records for a department with keys 'total' and 'incidence'.
    - department_name: str, the name of the department.
    """

    department_records = department.records
    department_name = department.name

    time = list(range(len(department_records['total'])))
    
    df = pd.DataFrame({
        'Time': time,
        'Total': department_records['total'],
        'Incidence': department_records['incidence']
    })

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='value', hue='variable', data=pd.melt(df, ['Time']))
    plt.title(f'Total and Incidence vs. Time for {department_name}')
    plt.ylabel('Count')
    plt.xlabel('Time (Unit)')
    plt.legend(title='Type')
    plt.show()

def visualize_movement_and_test_result(test_result_df, movement_df):
    movement_df['date'] = pd.to_datetime(movement_df['date'])
    test_result_df['time'] = pd.to_datetime(test_result_df['time'])

    # Merge test_result and movement DataFrames
    merged_df = pd.merge(movement_df, test_result_df, left_on=['id', 'date'], right_on=['id', 'time'], how='left')
    merged_df = merged_df.sort_values(['id', 'date'])

    # Create a dictionary to map test results to colors
    color_map = {
        # if not in hospital, than nan
        'In hospital, not tested': 1,
        0: 2, # tested, negative
        1: 3, # tested, colonized
        2: -1,# tested, only infected, which is impossible
        3: 4, # tested, colonized and infected
    }

    # Create a new column 'color' based on the test result and hospital status
    merged_df['color'] = merged_df.apply(lambda row: 
                                        color_map['In hospital, not tested'] if pd.isnull(row['result'])
                                         else color_map[row['result']], axis=1)

    # Create a pivot table for the heatmap
    pivot_df = merged_df.pivot_table(index='id', columns='date', values='color', aggfunc='first')

    return merged_df, pivot_df

def plot_CD(CD, x_range = slice(200), y_range = slice(200)):
    plt.figure(figsize=(10, 10)) 
    colors = ['black','white','green', 'yellow', 'red'] 
    cmap = ListedColormap(colors)
    plt.imshow(CD[x_range, y_range], cmap=cmap, interpolation='none')
    plt.colorbar(ticks=np.arange(len(colors)+1))
    plt.show()

def plot_data_CD(data_path, simulation, x_range = slice(200), y_range = slice(200)):
    """show the real lab result as condensed matrix
    """
    lab_cdiff_sorted = pd.read_csv(data_path)
    lab_cdiff_sorted['time'] = pd.to_datetime(lab_cdiff_sorted['time'])

    day_zero = pd.to_datetime('2019-11-1')
    tracker = np.array(simulation.patient_tracker)
    counts = Counter()
    data_CD = np.min([np.ones_like(simulation.real_CD), simulation.real_CD], axis=0)

    for i, row in lab_cdiff_sorted.iterrows():
        day = (row['time'] - day_zero).days + 1
        location = np.where(tracker[:,day] == row['id'])
        if len(location[0]) > 0:
            counts[row['result']] += 1
            data_CD[location[0][0], day] = row['result']

    plt.figure(figsize=(10, 10)) 
    colors = ['black','white','green', 'purple', 'red'] 
    cmap = ListedColormap(colors)
    plt.imshow(data_CD[x_range, y_range], cmap=cmap, interpolation='none')
    plt.colorbar(ticks=np.arange(len(colors)+1))
    plt.show()

# def plot_heatmap(pivot_df, x_range = slice(100), y_range = slice(100)):
#     pivot_np = pivot_df.to_numpy()
#     pivot_np = np.where(np.isnan(pivot_np), 0, pivot_np)

#     # plt.figure(figsize=(15,10))
#     plt.imshow(pivot_np[x_range,y_range], cmap='viridis', interpolation='nearest')  # 'cmap' controls the color map

#     plt.colorbar()

#     plt.title('Heatmap')
#     plt.xlabel('Days')
#     plt.ylabel('Patient')

#     plt.show()
