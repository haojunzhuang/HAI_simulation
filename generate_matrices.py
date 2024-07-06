import numpy as np
import os
from simulation.simulation import Simulation
from tqdm import tqdm
import torch

if __name__ == "__main__":
    movement_folder_path = "data_generation/generated_movements"
    output_folder_path = "data_generation/generated_matrices/"
    matrix_per_movement = 3
    start_index = 1
    end_index = 5
    width_cutoff = 400 # make sure same width

    observed_matrices = []
    real_matrices = []

    for i in tqdm(range(start_index, end_index+1)):
        filename = f"sw_3_{i}.pkl"

        # print(f"Simulating {filename}")
        movement_path = os.path.join(movement_folder_path, filename)
        
        # first round simulation to gather information (matrix size)
        simulation = Simulation(movement_data_path=movement_path, 
                            cleaned=True, initial_info=None, initial_patients=None,
                            uniform_alpha=0.20, uniform_beta=0.02, uniform_gamma=0.3,
                            uniform_delta=0.05, uniform_zeta=0.01, uniform_eta=0.005,
                            test=False,
                            )
        simulation.simulate(silent=True)
        
        # print(f'simulating {matrix_per_movement} rounds for {filename}...')
        # second round to generate matrix
        for i in range(1, matrix_per_movement+1):
            simulation.init_condensed_matrix(padding=None) # to make sure same height, using hardcoded height for each dep
            simulation.simulate(silent=True)

        # print(f"Generation Successful for {filename}")
        observed_matrices.append(np.stack([simulation.observed_CD_infection[:,:width_cutoff],
                                           simulation.observed_CD_movement[:,:width_cutoff]], axis=0))
        real_matrices.append(np.expand_dims(simulation.real_CD_infection[:,:width_cutoff], axis=0))
    
    observed_matrices = np.array(observed_matrices) # (n, 2, 738, 400)
    real_matrices = np.array(real_matrices) # (n, 1, 738, 400)

    observed_matrices = torch.tensor(observed_matrices, dtype=torch.int8)
    torch.save(observed_matrices, output_folder_path + "observed_matrices.pt")
    real_matrices = torch.tensor(real_matrices, dtype=torch.int8)
    torch.save(real_matrices, output_folder_path + "real_matrices.pt")
