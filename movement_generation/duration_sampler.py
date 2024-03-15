import pandas as pd
import random
import numpy as np

class duration_sampler:
    
    def __init__(self, path) -> None:

        def filter_multiple_admissions(df):
            admissions_df = df[df['from_department'] == 'ADMISSION']

            admission_counts = admissions_df.groupby('id').size()
            multiple_admissions = admission_counts[admission_counts > 1].index.tolist()

            filtered_df = df[~df['id'].isin(multiple_admissions)]
            excluded_patients_count = len(multiple_admissions)

            return filtered_df, excluded_patients_count

        def calculate_stay_duration(filtered_df):
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])

            duration_df = filtered_df.groupby('id')['date'].agg([min, max])
            duration_df['duration'] = (duration_df['max'] - duration_df['min']).dt.days
            duration_df = duration_df[duration_df['duration'] <= 100]

            return duration_df['duration']
        
        data = pd.read_csv(path)
        # Step 1: Filter out patients with multiple admissions
        filtered_df, excluded_count = filter_multiple_admissions(data)

        # Step 2: Calculate the stay duration for each patient
        durations = calculate_stay_duration(filtered_df)
        
        self.durations = durations
        
    def sample(self):
        return random.sample(list(self.durations), 1)[0]
    
class toy_duration_sampler:

    def __init__(self) -> None:
        return
    
    def sample(self, mean=1.3, sd=0.95):
        """
        Sample from a random log-normal distribution
        """
        return int(round(max(0,np.random.lognormal(mean, sd))))
    
class quick_duration_sampler:
    def __init__(self, duration_path: str) -> None:
        self.durations = pd.read_csv(duration_path)['duration']
    
    def sample(self, mean=1.3, sd=0.95):
        return random.sample(list(self.durations), 1)[0]