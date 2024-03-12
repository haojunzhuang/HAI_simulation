import pandas as pd
import random

class entry_sampler:
    
    def __init__(self, path) -> None:

        def calculate_num_daily_entries(df):
            admissions_df = df[df['from_department'] == 'ADMISSION']
            daily_entries = admissions_df.groupby(['date']).size()
            df = daily_entries.reset_index()
            df.columns = ['date', 'num_entries']
            df['date'] = pd.to_datetime(df['date'])
            df['weekday'] = df['date'].dt.day_name()
            return df
    
        data = pd.read_csv(path)
        daily_entries_by_weekday = calculate_num_daily_entries(data)

        weekend_entries = daily_entries_by_weekday[(daily_entries_by_weekday['weekday'] == "Saturday") |
                                                (daily_entries_by_weekday['weekday'] == "Sunday")]

        weekday_entries = daily_entries_by_weekday[(daily_entries_by_weekday['weekday'] != "Saturday") &
                                                (daily_entries_by_weekday['weekday'] != "Sunday")]
        
        self.weekend_entries = weekend_entries
        self.weekday_entries = weekday_entries
    
    def sample(self, date):
        weekday = pd.to_datetime(date).weekday()
        if weekday < 5:
            return random.sample(list(self.weekday_entries['num_entries']), 1)[0]
        else:
            return random.sample(list(self.weekend_entries['num_entries']), 1)[0]