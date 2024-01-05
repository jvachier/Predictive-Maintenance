import pandas as pd
import pickle

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class Loading_files:
    def load_save_df(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_telemetry = pd.read_csv("../src/data/PdM_telemetry.csv")
        df_machines = pd.read_csv("../src/data/PdM_machines.csv")
        df_failures = pd.read_csv("../src/data/PdM_failures.csv")
        df_errors = pd.read_csv("../src/data/PdM_errors.csv")
        df_maintenance = pd.read_csv("../src/data/PdM_maint.csv")

        dbfile_telemetry = open("./pickle_files/loading/telemetry", "ab")
        dbfile_machines = open("./pickle_files/loading/machines", "ab")
        dbfile_failures = open("./pickle_files/loading/failures", "ab")
        dbfile_errors = open("./pickle_files/loading/errors", "ab")
        dbfile_maint = open("./pickle_files/loading/maint", "ab")

        pickle.dump(df_telemetry, dbfile_telemetry)
        pickle.dump(df_machines, dbfile_machines)
        pickle.dump(df_failures, dbfile_failures)
        pickle.dump(df_errors, dbfile_errors)
        pickle.dump(df_maintenance, dbfile_maint)

        dbfile_telemetry.close()
        dbfile_machines.close()
        dbfile_failures.close()
        dbfile_errors.close()
        dbfile_maint.close()
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
        dbfile_telemetry = open("./pickle_files/loading/telemetry", "rb")
        dbfile_machines = open("./pickle_files/loading/machines", "rb")
        dbfile_failures = open("./pickle_files/loading/failures", "rb")
        dbfile_errors = open("./pickle_files/loading/errors", "rb")
        dbfile_maint = open("./pickle_files/loading/maint", "rb")

        df_telemetry = pickle.load(dbfile_telemetry)
        df_machines = pickle.load(dbfile_machines)
        df_failures = pickle.load(dbfile_failures)
        df_errors = pickle.load(dbfile_errors)
        df_maintenance = pickle.load(dbfile_maint)

        dbfile_telemetry.close()
        dbfile_machines.close()
        dbfile_failures.close()
        dbfile_errors.close()
        dbfile_maint.close()
        return (
            df_telemetry,
            df_machines,
            df_failures,
            df_errors,
            df_maintenance,
        )
