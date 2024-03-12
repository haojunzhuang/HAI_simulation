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
    
class toy_path_sampler:

    def __init__(self, num_departments) -> None:
        """
        Initialize a random transition matrix given number of departments
        """
        self.num_departments = num_departments
        def generate_transition_matrix(num_department):
            matrix_size = num_department + 2
            transition_matrix = np.zeros((matrix_size, matrix_size))

            departments = ['ADMISSION'] + [f'DEPT_{i}' for i in range(1, num_department + 1)] + ['DISCHARGE']

            # Set ADMISSION row: Random probabilities to other departments except to DISCHARGE and itself
            transition_matrix[0, 1:-1] = np.random.random(num_department)
            transition_matrix[0, 1:-1] /= transition_matrix[0, 1:-1].sum()  # Normalize to sum to 1

            # Fill the rest of the matrix with random probabilities
            for i in range(1, matrix_size - 1):  # Skip the last row (DISCHARGE)
                row_probs = np.random.random(matrix_size)
                row_probs[0] = 0  # No transitions from other departments back to ADMISSION
                row_probs[-1] = 0.1 + 0.2 * np.random.random()  # Random prob. to DISCHARGE, ensuring it's non-zero but not too high
                row_probs /= row_probs.sum()  # Normalize row to sum to 1
                transition_matrix[i] = row_probs

            transition_df = pd.DataFrame(transition_matrix, index=departments, columns=departments)
            return transition_df
        
        self.transition_matrix = generate_transition_matrix(self.num_departments)
    
    def sample(self, duration):
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
        return simulate_path(self.transition_matrix, num_step=duration)