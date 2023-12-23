import pandas as pd
import pickle


class Data_Preparation:
    def __init__(self, df: pd.DataFrame, name: str) -> None:
        self.data: pd.DataFrame = df
        self.feature: str = name

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
