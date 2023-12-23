import pandas as pd
import pickle


class Loading_files:
    def __init__(self) -> None:
        pass

    def load_save_df(self) -> pd.DataFrame:
        df_extruder_mass_temperature = pd.read_csv(
            "../data/extruder_mass_temperature_three_months.txt", sep="\t", header=None
        )

        df_extruder_mass_pressure = pd.read_csv(
            "../data/extruder_mass_pressure_two_months.txt", sep="\t", header=None
        )

        df_extruder_motor_speed = pd.read_csv(
            "../data/extruder_motor_speed_two_months.txt", sep="\t", header=None
        )

        df_extruder_motor_current = pd.read_csv(
            "../data/extruder_motor_current_two_months.txt", sep="\t", header=None
        )

        df_dosing_filling_on = pd.read_csv(
            "../data/dosing_filling_is_switched_on_six_months.txt",
            sep="\t",
            header=None,
        )

        dbfile_tmp = open("./pickle_files/loading/temperature", "ab")
        dbfile_pressure = open("./pickle_files/loading/pressure", "ab")
        dbfile_speed = open("./pickle_files/loading/speed", "ab")
        dbfile_current = open("./pickle_files/loading/current", "ab")
        dbfile_dosing_filling_on = open(
            "./pickle_files/loading/dosing_filling_on", "ab"
        )

        pickle.dump(df_extruder_mass_temperature, dbfile_tmp)
        pickle.dump(df_extruder_mass_pressure, dbfile_pressure)
        pickle.dump(df_extruder_motor_speed, dbfile_speed)
        pickle.dump(df_extruder_motor_current, dbfile_current)
        pickle.dump(df_dosing_filling_on, dbfile_dosing_filling_on)

        dbfile_pressure.close()
        dbfile_speed.close()
        dbfile_tmp.close()
        dbfile_current.close()
        dbfile_dosing_filling_on.close()
        return (
            df_extruder_mass_temperature,
            df_extruder_mass_pressure,
            df_extruder_motor_speed,
            df_extruder_motor_current,
            df_dosing_filling_on,
        )

    def load_db_file(self) -> pd.DataFrame:
        dbfile_df_tmp = open("./pickle_files/loading/temperature", "rb")
        dbfile_df_pressure = open("./pickle_files/loading/pressure", "rb")
        dbfile_df_speed = open("./pickle_files/loading/speed", "rb")
        dbfile_df_current = open("./pickle_files/loading/current", "rb")
        dbfile_df_dosing_filling_on = open(
            "./pickle_files/loading/dosing_filling_on", "rb"
        )

        df_extruder_mass_temperature = pickle.load(dbfile_df_tmp)
        df_extruder_mass_pressure = pickle.load(dbfile_df_pressure)
        df_extruder_motor_speed = pickle.load(dbfile_df_speed)
        df_extruder_motor_current = pickle.load(dbfile_df_current)
        df_dosing_filling_on = pickle.load(dbfile_df_dosing_filling_on)

        dbfile_df_tmp.close()
        dbfile_df_pressure.close()
        dbfile_df_speed.close()
        dbfile_df_current.close()
        dbfile_df_dosing_filling_on.close()
        return (
            df_extruder_mass_temperature,
            df_extruder_mass_pressure,
            df_extruder_motor_speed,
            df_extruder_motor_current,
            df_dosing_filling_on,
        )
