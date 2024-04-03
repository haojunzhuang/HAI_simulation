import numpy as np
import os
from simulation.simulation import Simulation
from tqdm import tqdm

if __name__ == "__main__":
    movement_folder_path = "data_generation/generated_movements"
    output_folder_path = "data_generation/generated_matrices/"
    matrix_per_movement = 1
    start_index = 300
    end_index = 500
    width_cutoff = 400 # make sure same width

    for i in tqdm(range(start_index, end_index+1)):
        filename = f"sw_3_{i}.pkl"
        if filename.endswith('.pkl'):
            try:
                # print(f"Simulating {filename}")
                movement_path = os.path.join(movement_folder_path, filename)
                
                # first round simulation to gather data
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
                    simulation.init_condensed_matrix(padding=None) # using hardcoded height for each dep
                    simulation.simulate(silent=True)

                    # save the matrix
                    full_matrix_path = output_folder_path + filename[:-4] + '_full_' + str(i)
                    np.save(full_matrix_path, simulation.real_CD[:,:width_cutoff])
                    partial_matrix_path =  output_folder_path + filename[:-4] + '_partial_'+ str(i)
                    np.save(partial_matrix_path, simulation.observed_CD[:,:width_cutoff])
                # print(f"Generation Successful for {filename}")
            except:
                print(f"Error in Generating for {filename}")
                continue
        # else:
        #     continue
        #     raise NotImplementedError()
