from typing import Any
from .patient import Patient
from .status import Status
import random
import time
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
        return  f"Department Name: {self.name}\n" + f"Number of Patients: {len(self.patients)}\n" + \
                f"Infected: {self.get_num_pos()}, Not Infected: {self.get_num_neg()}\n" + f"Info: [{info_str}]"
        

    def __str__(self) -> str:
        return self.verbose_info()

    def get_num_pos(self) -> int:
        """_summary_

        Returns
        -------
        int
            Number of colonized or infected Patients.
        """
        return sum([p.status == Status.colonized or p.status == Status.infected for p in self.patients])

    def get_num_neg(self) -> int:
        return len(self.patients) - self.get_num_pos()

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
        
        return "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +\
            "[Department]\n" +\
            f"Name: {self.name}\n" +\
            f"Total Patients: {len(self.patients)}\n" +\
            f"Patient Details:\n{patient_details}\n" +\
            f"Department Info: {self.info}\n" +\
            '---------------------------------\n'
        

    def accept_patient(self, patient:Patient):
        self.patients.add(patient)
    
    def release_patient(self, patient:Patient):
        self.patients.remove(patient)

    def infect(self, day:int, test=False, mode='contact'):
        """Handle infection and recovery

        Parameters
        ----------
        test : bool, optional
            _description_, by default False
        mode : str, optional
            Contact-based or contamination-based infection, by default 'contamination'
        """

        beta = self.info['beta']
        gamma = self.info['gamma']
        num_pos = self.get_num_pos()
        num_neg = self.get_num_neg()
        assert (num_pos + num_neg) == len(self.patients)

        if test:
            print(f"--------Start Infecting Department {self.name}--------\n")
            print( "Department Info: \n")
            print(f"Transmission Rate: {beta}")
            print(f"Recovery Rate: {gamma}")
            print(f"Number of Positive Patient: {num_pos}")
            print(f"Number of Negative Patient: {num_neg}\n")

        # Handle Empty Department
        if (num_pos + num_neg) == 0:
            return
        
        # Handle Recovery before Infection
        for patient in self.patients:
            if patient.status == Status.infected:
                if test:
                    print(f"Infected Patient: {patient}")
                r = random.random()
                if r < gamma:
                    patient.recover()
                    if test:
                        print(f"Get {r} Smaller Than {gamma}. Recovering Patient...")
                        print(f"{patient}")
        
        # Now Handle Infection
        num_infected = 0
        if mode == 'contact':
            p = min(beta * num_pos * num_neg / (num_pos + num_neg), 1)
            num_infected += np.random.binomial(self.get_num_neg(), p=p)
            if test:
                print(f"Probability of infection: {p}")
                print(f"Number of new patient infected: {num_infected}\n")
        elif mode == 'contamination':
            Ppl = self.info['Ppl']
            Plp = self.info['Plp']
            CD  = self.info['cleaning_duration']
            # 0 is not contaminated, 1 is contaminated, currently discrete currently discrete
            # either clean, or already contaminated, or newly contaminated with due to infected patient staying
            if test:
                print(f"Old Contamination: {self.contamination}")
            new_contamination = min(1, np.random.binomial(num_pos, Ppl))
            self.contamination = max(self.contamination, new_contamination)
            num_infected += np.random.binomial(num_neg, Plp)
            if test:
                print(f"New Contamination: {self.contamination}")
                print(f"Number of new patient infected: {num_infected}\n")

            # Cleaning the Department
            if day % CD == 0:
                self.contamination = 0
                if test:
                    print(f"Cleaning the Department at Day {day}...")

        else:
            raise NotImplementedError("Onlt contact and contamination mode are supported.")
        i = num_infected

        # "Shuffle the set"
        patient_list = list(self.patients)
        random.shuffle(patient_list)
        self.patients = set(patient_list)
        for patient in self.patients:
            if i == 0:
                break
            if not patient.status == Status.infected:
                i -= 1
                patient.infect()
                if test:
                    print(f"Patient {num_infected - i} being infected: {patient}")
        
        incidence = min(num_neg, num_infected)
        self.record(incidence=incidence)

        if test:
            print(f"Number of Incidence: {incidence}")
            print(f"--------End Infecting Department {self.name}--------\n")
            time.sleep(3)

    def develop(self, δ: float, test=False):
        """
        Develop symptoms.
        Only infected patients can develop symptoms.
        If patient is infected but not yet have symptom, develop symptoms with probability δ.
        If patient already has symptoms, increment the days by one.
        """

        for patient in self.patients:
            if patient.status == Status.infected:
                patient.develop_symptom(δ)

                if test:
                    print(f"\033[95mDeveloping Symptoms for Patient {patient.id}, Now: {patient.symptom}\033[0m")

    def surveil(self, test=False):
        """
        to be overriden
        """
        return self._surveil_everyone(test)

    def _surveil_everyone(self, test):
        """
        Naively surveil everyone in the department.
        """

        for patient in self.patients:
            result = patient.lab()

            if test:
                print(f'\033[96mTested patient {patient.id} and found {result}.\033[0m')



