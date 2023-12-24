import pandas as pd
import pickle


class Data_Preparation:
    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df3: pd.DataFrame,
        df5: pd.DataFrame,
    ) -> None:
        self.telemetry: pd.DataFrame = df1
        self.maintenance: pd.DataFrame = df2
        self.errors: pd.DataFrame = df3
        self.failures: pd.DataFrame = df5  # target

    def df_prepared(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(0)
        df.loc[df["failure"] != 0, "failure"] = 1
        df["failure"] = df["failure"].astype("int64")
        return df

    def merge_df(self) -> pd.DataFrame:
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
        self.telemetry["datetime"] = pd.to_datetime(self.telemetry["datetime"])
        self.maintenance["datetime"] = pd.to_datetime(self.maintenance["datetime"])
        self.errors["datetime"] = pd.to_datetime(self.errors["datetime"])
        self.failures["datetime"] = pd.to_datetime(self.failures["datetime"])

    def _merge1(self, features1: pd.DataFrame, feature2: pd.DataFrame) -> pd.DataFrame:
        merged_failure_maint = pd.merge(
            features1, feature2, how="left", on=["datetime", "machineID"]
        )
        return merged_failure_maint


class Load_Save:
    def __init__(self) -> None:
        pass

    def load_dataframe(self) -> pd.DataFrame:
        dbfile_dataframe = open("./pickle_files/data_preparation/data_set", "rb")
        data_set = pickle.load(dbfile_dataframe)
        dbfile_dataframe.close()
        return data_set

    def save_dataframe(self, data_set: pd.DataFrame) -> None:
        dbfile_dataframe = open("./pickle_files/data_preparation/data_set", "ab")
        pickle.dump(data_set, dbfile_dataframe)
        dbfile_dataframe.close()
