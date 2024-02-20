import pandas as pd
import tqdm
import time
import datetime
import random
from .department import Department
from .patient import Patient
from .utils.data_utils import (compress_by_day, compress_self_loop,
                               keep_departments_of_interest,
                               read_movement_data)

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
        uniform_alpha = 0.1, uniform_beta = 0.05, uniform_gamma = 0.1
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

        self.movements  = self.import_data(self, movement_data_path, cleaned=cleaned)

        self.node_names = self.movements.from_department.unique()
        self.node_names = [name for name in self.node_names if ((name != 'ADMISSION') and (name != 'DISCHARGE'))]

        if initial_patients and initial_info:
            assert set(self.node_names) == set(initial_patients.keys())
            assert set(self.node_names) == set(initial_info.keys())
            self.nodes = {name: Department(name, initial_patients[name], initial_info[name]) for name in self.node_names}
        else:
            self.nodes = {name: Department(name, [], {'alpha': uniform_alpha, 'beta': uniform_beta, 'gamma': uniform_gamma}) for name in self.node_names}

        self.record = {}

    @staticmethod
    def import_data(self, movement_data_path: str, cleaned: bool) -> pd.DataFrame:
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

        if cleaned:
            mvt = pd.read_csv(movement_data_path)
        else:
            # preprocessing steps
            mvt = read_movement_data(movement_data_path)
            mvt = compress_by_day(mvt)
            mvt = compress_self_loop(mvt)
            mvt = keep_departments_of_interest(mvt)
            mvt = mvt[(mvt['from_department'] != 'ADMISSION' ) | (mvt['to_department'] != 'DISCHARGE')] # handle the edge case of immediate discharge
            mvt = mvt.sort_values(by="date")

            mvt.to_csv(movement_data_path[:-4] + "_cleaned.csv")

        return mvt

    def export_checkpoint(self):
        patients = {name: self.nodes[name].patients for name in self.node_names}
        info     = {name: self.nodes[name].info     for name in self.node_names}
        return patients, info
    
    def move_patient(self, row):
        """
        For each row in the movement data, move the patient accordingly

        Parameters
        ----------
        row : pd.Dataframe
            one row of the movement data
        """

        def find_patient(id, dep):
            for patient in dep:
                if patient.id == id:
                    return patient
            raise NameError("Patient ID Not Found")
        
        # Handle Admission
        if row['from_department'] == 'ADMISSION':
            new_patient = Patient(row['id'], info={})
            if random.random() < self.nodes[row['to_department']].info['alpha']:
                new_patient.infect()
            self.nodes[row['to_department']].accept_patient(new_patient)

        else:
            current_patient = find_patient(row['id'], self.nodes[row['from_department']].patients)
            self.nodes[row['from_department']].release_patient(current_patient)
            if row['to_department'] != 'DISCHARGE':
                self.nodes[row['to_department']].accept_patient(current_patient)

    def update_patient_status(self):
        """
        Update patient status (perform infection and recovery process)
        """

        for _, dep in self.nodes.items():
            dep.infect()

    def simulate(self, timed = False):
        """
        Perform the simulation of the given movement data

        Parameters
        ----------
        timed : bool
            Whether to test the speed of the simulation or not
        """

        start_time = time.time()

        day = 0
        start_date = datetime.datetime.strptime(self.movements.iloc[0]['date'], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(self.movements.iloc[-1]['date'], "%Y-%m-%d")
        duration = (end_date - start_date).days
        current_date = start_date

        with tqdm.tqdm() as pbar:
            for _, row in self.movements.iterrows():
                self.move_patient(row)
                # Move patients first before infection
                while datetime.datetime.strptime(row['date'], "%Y-%m-%d") != current_date:
                    day += 1
                    current_date += datetime.timedelta(days=1)
                    pbar.set_description(f'Processing Day {day}/{duration}')
                    self.update_patient_status()
                
        end_time = time.time()
        time_taken = end_time - start_time

        if timed:
            print(f"Used {time_taken} for {self.movements.shape[0]} rows of movement and {duration} days")
    
