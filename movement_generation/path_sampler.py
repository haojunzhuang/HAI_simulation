import pandas as pd
import random
import numpy as np
import tqdm

class path_sampler:
    
    def __init__(self, data_path, transition_matrix_folder_path=None) -> None:
        data = pd.read_csv(data_path).drop(columns=['Unnamed: 0']).dropna()
        self.data = data
        if transition_matrix_folder_path:
            self.transition_matrix_folder_path = transition_matrix_folder_path

    def create_transition_matrices(self, method, max_duration = 100, window_size=None):
        def create_transition_matrix(data):
            departments = np.unique(data[['from_department', 'to_department']])

            transition_counts = pd.DataFrame(0, index=departments, columns=departments)

            for _, row in data.iterrows():
                transition_counts.at[row['from_department'], row['to_department']] += 1

            transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
            transition_matrix = transition_matrix.fillna(0)

            return transition_matrix
        
        def filter_patients_by_duration(df, k):
            df['date'] = pd.to_datetime(df['date'])
            
            duration_df = df.groupby('id')['date'].agg(['min', 'max']).reset_index()
            duration_df['duration'] = (duration_df['max'] - duration_df['min']).dt.days
            
            if method == "longer_only":
                # Filter out patients with duration less than k days
                filtered_ids = duration_df[(duration_df['duration'] >= k) & 
                                    (duration_df['duration'] <= max_duration+1)]['id']
            elif method == "shorter_only":
                # Filter out patients with duration more than k days
                filtered_ids = duration_df[duration_df['duration'] <= k]['id']
            elif method == "sliding_window":
                # Only include patients with duration in the range [k-cutoff_day, k+cutoff_day]
                filtered_ids = duration_df[(duration_df['duration'] >= max(0, k-window_size)) & 
                                    (duration_df['duration'] <= min(max_duration+1, k+window_size))]['id']
            else:
                raise ValueError("Method not Supported")
            
            filtered_df = df[df['id'].isin(filtered_ids)]
            
            return filtered_df
        
        for k in tqdm.tqdm(range(1, max_duration+1)):
            filtered_df = filter_patients_by_duration(self.data, k)
            transition_matrix = create_transition_matrix(filtered_df)
            if method == "sliding_window":
                transition_matrix.to_pickle(f"{self.transition_matrix_folder_path}/{k}_day_{method}_size_{window_size}.pkl")
            else:
                transition_matrix.to_pickle(f"{self.transition_matrix_folder_path}/{k}_day_{method}.pkl")

    def sample(self, duration, method, window_size=0):
        def simulate_path(transition_matrix, num_step=0, start_state='ADMISSION', end_state='DISCHARGE'):
            """
            Simulate a path through the hospital departments using the transition matrix.

            Parameters:
            - transition_matrix: pd.DataFrame - A transition matrix where rows and columns are departments,
                                                and values are transition probabilities.
            - start_state: str - The starting department for the simulation.
            - end_state: str - The department that ends the simulation.

            Returns:
            - path: list - A list of departments representing the simulated path.
            """
            current_state = start_state
            path = [current_state]

            if num_step > 0:
                for i in range(num_step):
                    probabilities = transition_matrix.loc[current_state]
                    assert np.isclose(probabilities.sum(),1), print(f"probabilities = {probabilities} does not sum to 1")

                    next_state = end_state
                    while next_state == end_state:
                        next_state = np.random.choice(probabilities.index, p=probabilities.values)
                    
                    path.append(next_state)
                    current_state = next_state
                path.append(end_state)
            else:
                while current_state != end_state:
                    probabilities = transition_matrix.loc[current_state]
                        
                    probabilities /= probabilities.sum()
                    assert np.isclose(probabilities.sum(),1), print(f"probabilities = {probabilities} does not sum to 1")

                    next_state = np.random.choice(probabilities.index, p=probabilities.values)

                    # Append the next state to the path and update the current state
                    path.append(next_state)
                    current_state = next_state

            return path
        
        assert duration >= 0, "Duration needs to be positive"
        if method == "sliding_window":
            transition_matrix = pd.read_pickle(f"{self.transition_matrix_folder_path}/{duration}_day_{method}_size_{window_size}.pkl")
        else:
            transition_matrix = pd.read_pickle(f"{self.transition_matrix_folder_path}/{duration}_day_{method}.pkl")
        return simulate_path(transition_matrix, num_step=duration)