import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from entry_sampler import entry_sampler
from duration_sampler import duration_sampler
from path_sampler import path_sampler

def compare_entry_distribution(real_movement, generated_movement):
    def report_entry_distribution(df, bins=50):
        entry_series = df['num_entries']
        mean_entry = entry_series.mean()
        std_entry = entry_series.std()

        print(f"Mean Number of Entries Per Day: {mean_entry}")
        print(f"Standard Deviation: {std_entry}")

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        entry_series.hist(bins=bins, edgecolor='k')
        plt.title('Distribution of Number of Daily Entries')
        plt.xlabel('Number of Entries')
        plt.ylabel('Frequency')
        #plt.xlim(0,50)
        plt.grid(False)
        plt.show()

    report_entry_distribution(real_movement['num_entries'])