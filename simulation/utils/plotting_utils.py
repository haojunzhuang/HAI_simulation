from ..department import Department
from ..simulation import Simulation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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