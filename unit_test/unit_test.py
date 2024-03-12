from simulation.department import Department
from simulation.patient import Patient
from simulation.simulation import Simulation

def perform_test(path, alpha, beta, gamma):
    
    simulation = Simulation(movement_data_path=path, 
                        cleaned=True, initial_info=None, initial_patients=None,
                        uniform_alpha=alpha, uniform_beta=beta, uniform_gamma=gamma,
                        test=True)
    simulation.simulate()

# Test 1: No Infection Spread
# Test 2: Rapid Patient Turnover
# Test 3: Long Range Infection

# test_ID = 3

# perform_test(path=f"simulation/test_data/test_dataset_{test_ID}.csv",
#              alpha=0.8, beta=0.5, gamma=0.8)

Simulation.import_data(None, 'data/movements.csv', cleaned=False, fill=True, remove_loop=False)