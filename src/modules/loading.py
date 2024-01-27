from typing import Tuple
from dataclasses import dataclass
import pickle
import pandas as pd


@dataclass(slots=True)
class Loading_files:
    """
    Class to load and save required data into a pickle format
    """

    def load_save_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Saving csv files into pickle ones.

        Return: 5 dataframes for each required files
        """
        df_telemetry = pd.read_csv("../src/data/PdM_telemetry.csv")
        df_machines = pd.read_csv("../src/data/PdM_machines.csv")
        df_failures = pd.read_csv("../src/data/PdM_failures.csv")
        df_errors = pd.read_csv("../src/data/PdM_errors.csv")
        df_maintenance = pd.read_csv("../src/data/PdM_maint.csv")

        with open("./pickle_files/loading/telemetry", "ab") as dbfile_telemetry:
            pickle.dump(df_telemetry, dbfile_telemetry)
        with open("./pickle_files/loading/machines", "ab") as dbfile_machines:
            pickle.dump(df_machines, dbfile_machines)
        with open("./pickle_files/loading/failures", "ab") as dbfile_failures:
            pickle.dump(df_failures, dbfile_failures)
        with open("./pickle_files/loading/errors", "ab") as dbfile_errors:
            pickle.dump(df_errors, dbfile_errors)
        with open("./pickle_files/loading/maint", "ab") as dbfile_maint:
            pickle.dump(df_maintenance, dbfile_maint)

        return (
            df_telemetry,
            df_machines,
            df_failures,
            df_errors,
            df_maintenance,
        )

    def load_db_file(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loading pickle files.

        Return: 5 dataframes for each required files
        """
        with open("./pickle_files/loading/telemetry", "rb") as dbfile_telemetry:
            df_telemetry = pickle.load(dbfile_telemetry)
        with open("./pickle_files/loading/machines", "rb") as dbfile_machines:
            df_machines = pickle.load(dbfile_machines)
        with open("./pickle_files/loading/failures", "rb") as dbfile_failures:
            df_failures = pickle.load(dbfile_failures)
        with open("./pickle_files/loading/errors", "rb") as dbfile_errors:
            df_errors = pickle.load(dbfile_errors)
        with open("./pickle_files/loading/maint", "rb") as dbfile_maint:
            df_maintenance = pickle.load(dbfile_maint)

        return (
            df_telemetry,
            df_machines,
            df_failures,
            df_errors,
            df_maintenance,
        )
