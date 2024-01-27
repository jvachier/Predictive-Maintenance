from dataclasses import dataclass
import pickle
import pandas as pd


@dataclass(slots=True)
class Data_Preparation:
    """
    Class to prepare data to be ingested by the model
    Input:
        - telemetry: pd.DataFrame
        - maintenance: pd.DataFrame
        - errors: pd.DataFrame
        - failures: pd.DataFrame

    Four functions:
        - df_prepared
        - merge_df
        - date_to_time
        - _merge1
    """

    telemetry: pd.DataFrame
    maintenance: pd.DataFrame
    errors: pd.DataFrame
    failures: pd.DataFrame  # target

    def df_prepared(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to prepare 'Target' - failure
        Input:
            - df: pd.DataFrame
        Return:
            - df: pd.DataFrame
        """
        df = df.fillna(0)
        df.loc[df["failure"] != 0, "failure"] = 1
        df["failure"] = df["failure"].astype("int64")
        return df

    def merge_df(self) -> pd.DataFrame:
        """
        Function combining multiple data into one main dataframe
        Return:
            - merge_df_feature_target: pd.DataFrame
        """
        maintenance_dummies = pd.get_dummies(
            self.maintenance, columns=["comp"], drop_first=True, dtype=int
        )
        errors_dummies = pd.get_dummies(
            self.errors, columns=["errorID"], drop_first=True, dtype=int
        )

        merged_failure_maint = self._merge1(self.telemetry, maintenance_dummies)
        merged_failure_maint_error = self._merge1(merged_failure_maint, errors_dummies)

        merge_df_feature_target = pd.merge(
            merged_failure_maint_error,
            self.failures,
            how="left",
            on=["datetime", "machineID"],
        )
        return merge_df_feature_target

    def date_to_time(self) -> None:
        """
        Function time into datetime.
        """
        self.telemetry["datetime"] = pd.to_datetime(self.telemetry["datetime"])
        self.maintenance["datetime"] = pd.to_datetime(self.maintenance["datetime"])
        self.errors["datetime"] = pd.to_datetime(self.errors["datetime"])
        self.failures["datetime"] = pd.to_datetime(self.failures["datetime"])

    def _merge1(self, features1: pd.DataFrame, feature2: pd.DataFrame) -> pd.DataFrame:
        """
        Function combining multiple data into one main dataframe
        Return:
            - merged_failure_maint: pd.DataFrame
        """
        merged_failure_maint = pd.merge(
            features1, feature2, how="left", on=["datetime", "machineID"]
        )
        return merged_failure_maint


@dataclass(slots=True)
class Load_Save:
    """
    Class to load and save the prepared data into a pickle format: data_set
    Two functions:
        - load_dataframe
        - save_dataframe
    """

    def load_dataframe(self) -> pd.DataFrame:
        """
        Function loading prepared data: data_set.

        Return:
            - data_set: pd.DataFrame
        """
        with open("./pickle_files/data_preparation/data_set", "rb") as dbfile_dataframe:
            data_set = pickle.load(dbfile_dataframe)
        return data_set

    def save_dataframe(self, data_set: pd.DataFrame) -> None:
        """
        Function saving prepared data into pickle file: data_set.
        """
        with open("./pickle_files/data_preparation/data_set", "ab") as dbfile_dataframe:
            pickle.dump(data_set, dbfile_dataframe)
