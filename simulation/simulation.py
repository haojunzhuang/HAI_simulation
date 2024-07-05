import pandas as pd
import numpy as np
import tqdm
import time
import datetime
import random
from .department import Department
from .patient import Patient
from .status import Status
from .utils.data_utils import (compress_by_day, compress_self_loop,
                               keep_departments_of_interest,
                               read_movement_data)
from collections import deque

# CD_movement  ----- 0: none; 1: admission; 2:stay; 3: transfer; 4: discharge
# CD_infection ----- 0: none; 1: unkown; 2: tested negative; 3: tested colonized 4: tested infected

class Simulation:
    """
    Network ABM simulation for HAI transmission.
    """

    def __init__(
        self,
        movement_data_path: str,
        cleaned: bool,
        initial_patients: dict[str, list[Patient]] | None,
        initial_info: dict[str, dict] | None,
        uniform_alpha = 0.1, uniform_beta = 0.05, uniform_gamma = 0.1, 
        uniform_delta = 0.15, uniform_zeta = 0.05, uniform_eta = 0.20,
        test = False, dummy_transition = False
    ) -> None:
        """_summary_

        Parameters
        ----------
        movement_data_path : str
            Path to raw movement data csv.
        cleaned : bool
            _description_
        initial_patients : dict[str, list[Patient]]
            _description_
        initial_info : dict[str, dict]
            _description_
        """
        # TODO: Modify Docstring
        self.test = test
        self.dummy_transition = dummy_transition
        self.movements  = self.import_data(self, movement_data_path, cleaned=cleaned)
        self.initial_patients = initial_patients
        self.initial_info = initial_info
        self.uniform_alpha = uniform_alpha
        self.uniform_beta = uniform_beta
        self.uniform_gamma = uniform_gamma
        self.uniform_delta = uniform_delta
        self.uniform_zeta = uniform_zeta
        self.uniform_eta = uniform_eta

        self.condensed_matrix_mode = False
        self.real_lab_data_mode = False

    def setup(self):
        if self.test:
            print("Setting Up Simulation...")
            time.sleep(2)
        
        self.record = {}

        self.node_names = self.movements.from_department.unique()
        self.node_names = [name for name in self.node_names if ((name != 'ADMISSION') and (name != 'DISCHARGE'))]
        self.node_names.sort()

        if self.initial_patients and self.initial_info:
            assert set(self.node_names) == set(self.initial_patients.keys())
            assert set(self.node_names) == set(self.initial_info.keys())
            self.nodes = {name: Department(name, self.initial_patients[name], self.initial_info[name]) for name in self.node_names}
        else:
            self.nodes = {name: Department(name, [], {'alpha': self.uniform_alpha, 'beta': self.uniform_beta, 
                                                      'gamma': self.uniform_gamma}) for name in self.node_names}
        
        if self.test:
            for name in self.nodes:
                print(self.nodes[name])
        
        if self.dummy_transition:
            self.dummy_pool = []

    @staticmethod
    def import_data(self, movement_data_path: str, cleaned: bool, fill=False, remove_loop=True) -> pd.DataFrame:
        """
        If the data is not cleaned, preprocess data and save it in a cleaned version.
        If the data is cleaned, import data directly.
        Can be called static to clean data.

        Parameters
        ----------
        movement_data_path : str
            _description_
        cleaned : bool
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """

        if not cleaned:
            # preprocessing steps
            mvt = read_movement_data(movement_data_path, fill=fill)
            mvt = compress_by_day(mvt)
            if remove_loop: 
                mvt = compress_self_loop(mvt)
            mvt = keep_departments_of_interest(mvt)
            mvt = mvt[(mvt['from_department'] != 'ADMISSION' ) | (mvt['to_department'] != 'DISCHARGE')] # handle the edge case of immediate discharge
            mvt = mvt.sort_values(by=['id'], key=lambda x: x != 'DISCHARGE')

            movement_data_path = movement_data_path[:-4] + "_cleaned.csv"
            if fill: 
                movement_data_path = movement_data_path[:-4] + "_filled.csv"
            mvt.to_csv(movement_data_path)

        # return pd.read_csv(movement_data_path, index_col=0, parse_dates=False)
        if movement_data_path.endswith('.csv'):
            result = pd.read_csv(movement_data_path, parse_dates=False)
        else:
            result = pd.read_pickle(movement_data_path)
            result['date'] = result['date'].dt.strftime('%Y-%m-%d')
        
        result = result.sort_values(by=['date', 'from_department'], key = lambda x: x == 'ADMISSION' if x.name == 'from_department' else x)
        return result

    def export_checkpoint(self):
        patients = {name: self.nodes[name].patients for name in self.node_names}
        info     = {name: self.nodes[name].info     for name in self.node_names}
        if self.test:
            print("Exporting Checkpoint...")
            print(patients)
            print(info)
        return patients, info
    
    def init_condensed_matrix(self, padding):
        """
        called on second run, after completing the first simulation

        Initialize a condensed matrix representation analogous to bed management in hospital
        Hopefully convenient for Masked Autoencoder

        If no padding specified, then use hardcoded value (approximately padding=5)
        """
        PADDING = padding
        
        self.condensed_matrix_mode = True
        assert self.total_days, "Needs to be called on second run"
        width = self.total_days
        
        if not padding:
            self.dep_sizes = {'10LS CVT': 80, '13L GEN SURG': 60, 'EMERGENCY DEPT PARN': 50, 
                              '8L NEUROSCIENCES': 60, '12M MED/SURG/ACUTE TCU': 30, '7E MED/SURG': 30, 
                              '7L MUSCULOSKELETAL': 60, '12L MEDSURG-ONC/BMT A': 55, '13I M/S ICU': 30, 
                              '10NE CARD ICU': 30, '15L ADULT ACUTE CARE': 55, '9L TRANSPLANT': 65, 
                              '8 NICU': 25, '11NE NICU': 30, '14L MEDICINE': 65, '6L NEUR TRAN': 50, 
                              'PPU': 25, '11L MEDSURG-ONC/BMT B': 55, '14M MS-HI-ACUITY': 50, 
                              '9NE M/S ICU': 30, '6ICC': 25, '8S TCU': 25, 'PERIOP PARN': 15}
        else:
            self.dep_sizes = {name: PADDING+max(self.nodes[name].records['total']) for name in self.node_names}
        
        height = sum(self.dep_sizes.values())
        # print(height)
        self.dep_start_pos = {name: sum(self.dep_sizes[self.node_names[i]] 
                            for i in range(self.node_names.index(name))) for name in self.node_names}
        # print(self.dep_sizes)
        self.bed_queues = {name: deque([i for i in range(self.dep_start_pos[name], self.dep_start_pos[name]+self.dep_sizes[name])])
                            for name in self.node_names}
        # print(self.bed_queues)

        # condensed matrices
        self.real_CD_infection = np.zeros((height, width))
        self.observed_CD_infection = np.zeros((height, width))
        self.observed_CD_movement = np.zeros((height, width))
        self.patient_tracker = [[None] * width for _ in range(height)]


    def allocate_bed(self, from_dep, to_dep, patient):
        """bed queue of each department
        """
        if from_dep != 'ADMISSION':
            self.bed_queues[from_dep].appendleft(patient.location)

        if to_dep != 'DISCHARGE':
            try:
                patient.location = self.bed_queues[to_dep].popleft()
            except Exception as e:
                print(e, f'allocation failed for {to_dep}')
                raise ValueError()

        
    
    def move_patient(self, row, day):
        """
        For each row in the movement data, move the patient accordingly

        Parameters
        ----------
        row : pd.Dataframe
            one row of the movement data
        day: int
            current day of the simulation
        """

        def find_patient(id, dep):
            for patient in dep:
                if patient.id == id:
                    return patient
            raise NameError("Patient ID Not Found")
        
        if self.test:
            print(row)
            print()

        # Handle Admission
        if row['from_department'] == 'ADMISSION':
            new_patient = Patient(row['id'], info={})
            if random.random() < self.nodes[row['to_department']].info['alpha']:
                new_patient.colonize()
            if self.dummy_transition:
                self._accept_dummy(self.nodes[row['to_department']], new_patient)
            else:
                self.nodes[row['to_department']].accept_patient(new_patient)
            if self.test:
                print(f"New Patient Entering: {new_patient} with status {new_patient.status}\n")
            patient = new_patient

        # Handle Transfer
        else:
            current_patient = find_patient(row['id'], self.nodes[row['from_department']].patients)
            if self.test:
                print(f"Before moving: {current_patient}")
            self.nodes[row['from_department']].release_patient(current_patient)
            if row['to_department'] != 'DISCHARGE':
                if self.dummy_transition:
                    self._accept_dummy(self.nodes[row['to_department']], current_patient)
                else:
                    self.nodes[row['to_department']].accept_patient(current_patient)
            if self.test:
                print(f"After moving: {current_patient}")
            patient = current_patient

        if self.condensed_matrix_mode:
            self.allocate_bed(row['from_department'], row['to_department'], patient)

            if row['from_department'] == 'ADMISSION':
                self.observed_CD_movement[patient.location, day] = 1
            elif row['to_department'] == 'DISCHARGE':
                self.observed_CD_movement[patient.location, day] = 4
            else: # transfer
                self.observed_CD_movement[patient.location, day] = 3

    def _accept_dummy(self, to_department: Department, patient: Patient):
        """
        Temperorily story transitioning patients
        """
        self.dummy_pool.append((to_department, patient))

    def _release_dummies(self):
        for to_department, patient in self.dummy_pool:
            to_department.accept_patient(patient)
        self.dummy_pool = []

    def update_patient_status(self, day: int, date: datetime.datetime):
        """
        Update patient status (perform infection and recovery process)
        Also, develop symptoms and surveil the department.
        """

        for _, dep in self.nodes.items():
            # Colonize and Recover
            dep.infect(day, test = self.test)
            # Infect and Symptom
            dep.develop(self.uniform_eta, self.uniform_delta, self.uniform_zeta, test=self.test)

            # Lab Test Record
            # real_results, observed_results = dep.surveil(test = self.test)
            # self._record_lab_results(date, self.real_lab_record, real_results)
            # self._record_lab_results(date, self.observed_lab_record, observed_results)
            if self.condensed_matrix_mode:
                observed_results = dep.surveil(test = self.test)
                
                for patient in dep.patients:
                    self.real_CD_infection[patient.location, day] = patient.status.value
                    self.observed_CD_infection[patient.location, day] = 1

                    self.patient_tracker[patient.location][day] = patient.id
                    if self.observed_CD_movement[patient.location, day] != 1: # if not admission
                        self.observed_CD_movement[patient.location, day] = 2  # then patient continue to stay
                
                for patient, status in observed_results.items():
                    self.observed_CD_infection[patient.location, day] = status.value

                # 0: none; 2: tested negative; 3: tested colonized 4: tested infected

    def simulate(self, timed = False, silent=False):
        """
        Perform the simulation of the given movement data

        Parameters
        ----------
        timed : bool
            Whether to test the speed of the simulation or not
        """
        self.setup()

        if self.test:
            print( "Starting Simulation...\n")
            time.sleep(3)

        start_time = time.time()

        day = 0
        start_date = datetime.datetime.strptime(self.movements.iloc[0]['date'], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(self.movements.iloc[-1]['date'], "%Y-%m-%d")
        duration = (end_date - start_date).days
        current_date = start_date

        if self.test:
            print(f"Simulation Start Date: {start_date}")
            print(f"Simulation End Date: {end_date}")
            print(f"Simulation Duration: {duration}\n")

        if silent:
            for index, row in self.movements.iterrows():
                if self.dummy_transition and datetime.datetime.strptime(row['date'], "%Y-%m-%d") != current_date:
                    self._release_dummies()

                self.move_patient(row, day) # Move patients first
                
                while datetime.datetime.strptime(row['date'], "%Y-%m-%d") != current_date:
                    day += 1
                    current_date += datetime.timedelta(days=1)

                    if self.dummy_transition:
                        self._release_dummies()

                    self.update_patient_status(day, current_date) # Then update patient status
        else:
            with tqdm.tqdm() as pbar:
                for index, row in self.movements.iterrows():

                    if self.test:
                        print(f"--------Reading Row {index}-------- \n")

                    if self.dummy_transition and datetime.datetime.strptime(row['date'], "%Y-%m-%d") != current_date:
                        self._release_dummies()

                    self.move_patient(row, day) # Move patients first

                    if self.test:
                        print(f"Current row is at {row['date']}, Day {(datetime.datetime.strptime(row['date'], '%Y-%m-%d') - start_date).days}")
                        print(f"--------Finish Reading Row {index}-------- \n")
                        time.sleep(3)

                    while datetime.datetime.strptime(row['date'], "%Y-%m-%d") != current_date:
                        day += 1
                        current_date += datetime.timedelta(days=1)

                        if self.test:
                            print(f"--------Processing Day {day}, {current_date}--------\n")
                        pbar.set_description(f'Processing Day {day}/{duration}')

                        self.update_patient_status(day, current_date) # Then update patient status

                        if self.test:
                            print(f"--------Finish Processing Day {day}, {current_date}--------\n")
                            time.sleep(3)

            
        self.total_days = day+1 # for init condensed matrix in second run

        if self.test:
            print("---------Simulation END---------\n")
        end_time = time.time()
        time_taken = end_time - start_time

        if timed:
            print(f"Used {time_taken} for {self.movements.shape[0]} rows of movement and {duration} days")

    def _record_lab_results(self, current_date, record: list[dict], results: dict[Patient, Status]):
        """
        Record the lab results of the patients in the department.

        Parameters
        ----------
        day : int
            The current day of the simulation
        results : dict[Patient, int]
            The results of the lab tests
        """
        pass
    
    def lab_results_to_df(self, option: str = 'real') -> pd.DataFrame:
        """
        Convert the lab results to a DataFrame

        Returns
        -------
        pd.DataFrame
            The lab results in a DataFrame
        """
        pass
    
