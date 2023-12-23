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
        self.failures: pd.DataFrame = df5 # target

    def merge_df(self) -> pd.DataFrame:
        
        merged_failure_maint = Data_Preparation._merge1()

        merge_df_feature_target = pd.merge(
            merged_failure_maint, self.failures, 
            how="left", 
            on=["datetime", "machineID"]
        )

        return df_prepared


    def _merge1(self) -> pd.DataFrame:
        merged_failure_maint = pd.merge(
            self.telemetry, self.maintenance, 
            how="left", 
            on=["datetime", "machineID"]
        ) 
        return merged_failure_maint

    def _date_to_time(self) -> None:
        self.telemetry["datetime"] = pd.to_datetime(self.telemetry["datetime"])
        self.maintenance["datetime"] = pd.to_datetime(self.maintenance["datetime"])
        self.errors["datetime"] = pd.to_datetime(self.errors["datetime"])
        self.failures["datetime"] = pd.to_datetime(self.failures["datetime"])


    def _feature_preparation(self):

    def data_structure(self) -> None:
        self.data[1] = self.data[1].str.replace(",", ".").astype(float)

    def data_time(self) -> pd.DataFrame:
        self.data = self.data[[0, 1]]
        self.data = self.data.rename(columns={0: "Time", 1: str(self.feature)})
        self.data["Time"] = pd.to_datetime(self.data["Time"])
        return self.data


class Load:
    def __init__(self) -> None:
        pass

    def load_dataframe(self) -> pd.DataFrame:
        dbfile_dataframe = open("./pickle_files/data_preparation/data_set", "rb")
        data_set = pickle.load(dbfile_dataframe)
        dbfile_dataframe.close()
        return data_set


class Data_Set:
    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df3: pd.DataFrame,
        df4: pd.DataFrame,
        df5: pd.DataFrame,
    ) -> None:
        self.feature1: pd.DataFrame = df1
        self.feature2: pd.DataFrame = df2
        self.feature3: pd.DataFrame = df3
        self.feature4: pd.DataFrame = df4
        self.target: pd.DataFrame = df5

    def merge(self) -> pd.DataFrame:
        merged_dataframe = pd.merge_asof(
            self.feature1.sort_values("Time"),
            self.feature2.sort_values("Time"),
            on="Time",
        )

        merged_dataframe2 = pd.merge_asof(
            merged_dataframe.sort_values("Time"),
            self.feature3.sort_values("Time"),
            on="Time",
        )

        merged_dataframe3 = pd.merge_asof(
            merged_dataframe2.sort_values("Time"),
            self.feature4.sort_values("Time"),
            on="Time",
        )

        data_set = pd.merge_asof(
            merged_dataframe3.sort_values("Time"),
            self.target.sort_values("Time"),
            on="Time",
        )
        data_set = data_set.dropna()
        return data_set

    def save_dataframe(self, data_set: pd.DataFrame) -> None:
        dbfile_dataframe = open("./pickle_files/data_preparation/data_set", "ab")
        pickle.dump(data_set, dbfile_dataframe)
        dbfile_dataframe.close()
