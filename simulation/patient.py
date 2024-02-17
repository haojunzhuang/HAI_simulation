from typing import Any


class Patient:
    """
    A patient agent.
    """

    def __init__(self, id: str, info: dict[str, Any]) -> None:
        """_summary_

        Parameters
        ----------
            id (str): 
                Patient ID (e.g. deid_enc_id).
            info (dict[str, Any]): 
                Fixed patient informations like demographics.
        """

        self.id = id
        self.info = info
        self.infected  = False
        self.recovered = False

    def infect(self) -> None:
        """
        Infect a patient.
        """

        self.infected = True

    def recover(self) -> None:
        """
        Recover a patient.
        """

        self.infected = False
        self.recovered = True
