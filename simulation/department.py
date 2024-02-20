from typing import Any
from .patient import Patient
import random
import numpy as np

class Department:
    """
    A department object.
    """

    def __init__(self, name: str, patients: list[Patient], info: dict[str, Any]) -> None:
        """_summary_

        Parameters
        ----------
        name : str
            Department name.
        patients : list[Patient]
            Set of patient instances.
        info : dict[str, Any]
            Department level informations, including parameters.
        """

        self.name = name
        self.patients = set(patients)
        self.info = info
        self.records = {'total': [], 'incidence': [], 'prevalence': []}
    
    def __repr__(self) -> str:
        info_str = ', '.join(f"{key}: {value}" for key, value in self.info.items())
        return (
            f"Department Name: {self.name}\n"
            f"Number of Patients: {len(self.patients)}\n"
            f"Infected: {self.get_num_pos()}, Not Infected: {self.get_num_neg()}\n"
            f"Info: [{info_str}]"
        )

    def __str__(self) -> str:
        return self.verbose_info()

    def get_num_pos(self) -> int:
        return sum([p.infected for p in self.patients])

    def get_num_neg(self) -> int:
        return sum([not p.infected for p in self.patients])

    def record(self, incidence: int) -> None:
        """
        Record prevalence, incidence, and num_patients of the department on a day.
        Incidence(number of new infections) needs to be provided and the others will be inferred.

        Parameters
        ----------
        incidence : int
            Number of new infections that day.
        """
        prevalence = self.get_num_pos()

        self.records['total'].append(len(self.patients))
        self.records['incidence'].append(incidence)
        self.records['prevalence'].append(prevalence)
    
    def verbose_info(self) -> str:
        """_summary_

        Returns
        -------
        str
            _description_
        """

        patient_details = '\n'.join([' ' + str(patient) for patient in self.patients])
        
        print (
            '+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n'
            f"[Department]\n"
            f"Name: {self.name}\n"
            f"Total Patients: {len(self.patients)}\n"
            f"Patient Details:\n{patient_details}\n"
            f"Department Info: {self.info}\n"
            '---------------------------------\n'
        )

    def accept_patient(self, patient:Patient):
        self.patients.add(patient)
    
    def release_patient(self, patient:Patient):
        self.patients.remove(patient)

    def infect(self):
        beta = self.info['beta']
        gamma = self.info['gamma']
        num_pos = self.get_num_pos()
        num_neg = self.get_num_neg()

        # Handle Empty Department
        if (num_pos + num_neg) == 0:
            return
        
        # Handle Recovery before Infection
        for patient in self.patients:
            if patient.infected:
                if random.random() < gamma:
                    patient.recover()
        
        p = min(beta * num_pos * num_neg / (num_pos + num_neg), 1)
        num_infected = np.random.binomial(self.get_num_neg(), p=p)
        i = num_infected
        # "Shuffle the set"
        patient_list = list(self.patients)
        random.shuffle(patient_list)
        self.patients = set(patient_list)

        for patient in self.patients:
            if i == 0:
                break
            if not patient.infected:
                i -= 1
                patient.infect()
        
        incidence = min(num_neg, num_infected)
        self.record(incidence=incidence)





