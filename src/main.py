import os.path
import pandas as pd
from keras.models import load_model

# modules / classes
import modules.data_preparation as data_preparation
import modules.models as models
import modules.loading as loading


def main():
    # load data
    print("Loading data\n")

    LOAD = loading.Loading_files()

    if os.path.isfile("./pickle_files/loading/telemetry") is False:
        (
            df_telemetry,
            df_machines,
            df_failures,
            df_errors,
            df_maintenance,
        ) = LOAD.load_save_df()
    else:
        (
            df_telemetry,
            df_machines,
            df_failures,
            df_errors,
            df_maintenance,
        ) = LOAD.load_db_file()

    quit()
    data_tmp = data_preparation.Data_Preparation(
        df_extruder_mass_temperature, "Temperature(C)"
    )

    data_pressure = data_preparation.Data_Preparation(
        df_extruder_mass_pressure, "Pressure(bar)"
    )

    data_target = data_preparation.Data_Preparation(df_dosing_filling_on, "Target")

    data_speed = data_preparation.Data_Preparation(
        df_extruder_motor_speed, "Speed(r/min)"
    )
    data_current = data_preparation.Data_Preparation(
        df_extruder_motor_current, "Current(A)"
    )

    set_load = data_preparation.Load()

    if os.path.isfile("./pickle_files/data_preparation/data_set") is False:
        data_speed.data_structure()
        data_current.data_structure()

        tmp = data_tmp.data_time()
        pressure = data_pressure.data_time()
        speed = data_speed.data_time()
        current = data_current.data_time()
        target = data_target.data_time()

        data_set = data_preparation.Data_Set(
            tmp,
            pressure,
            speed,
            current,
            target,
        )

        Set = data_set.merge()
        data_set.save_dataframe(Set)

    else:
        Set = set_load.load_dataframe()

    # models

    anomaly_on = "False"
    if anomaly_on != "True":
        print("Models prediction\n")
        model = models.Predictions(Set)

        save_model = models.Save_Load_models()

        X_train, X_test, y_train, y_test = model.train_split()
        if os.path.isfile("./pickle_files/models/lr") is False:
            pipe_lr, y_predic_lr, y_predic_lr_proba = model.model_lr(
                X_train, y_train, X_test
            )
            clf_RFC, y_pred_RFC, y_pred_RFC_proba = model.model_RF(
                X_train, y_train, X_test
            )
            save_model.save_model_sklearn("lr", pipe_lr, y_predic_lr, y_predic_lr_proba)
            save_model.save_model_sklearn("rf", clf_RFC, y_pred_RFC, y_pred_RFC_proba)
        else:
            pipe_lr, y_predic_lr, y_predic_lr_proba = save_model.load_model_sklearn(
                "lr"
            )
            clf_RFC, y_pred_RFC, y_pred_RFC_proba = save_model.load_model_sklearn("rf")

        model.model_metrics(y_test, y_predic_lr_proba, "Logictic Regression")
        model.model_metrics(y_test, y_pred_RFC_proba, "Random Forest")

        model.roc_curve(pipe_lr, clf_RFC, X_test, y_test)
        model.visualization_prediction(y_test, y_pred_RFC, "RF")

        model.visualization_accuracy(pipe_lr, "Logistic Regression", X_train, y_train)
        model.visualization_accuracy(clf_RFC, "Random Forest", X_train, y_train)
    else:
        print("\n")
        print("Models Anomaly Detection\n")
        tmp = data_tmp.data_time()
        pressure = data_pressure.data_time()

        anomaly_isolation = models.Anomaly_detection_isolationforest(
            pressure, "Pressure(bar)"
        )
        anomaly_autoencoder = models.Anomaly_detection_autoencoder(
            pressure, "Pressure(bar)"
        )

        anomaly_isolation.isolationforest()
        anomaly_isolation.visulaization_isolationforest()
        x_train = anomaly_autoencoder.data_to_feed_autoencoder()
        if os.path.isfile("./pickle_files/models/autoencoder.keras") is False:
            autoencoder = anomaly_autoencoder.AutoEncoder(x_train)
            mse = anomaly_autoencoder.result_autocendoer(autoencoder, x_train)
            anomaly_autoencoder.Anomaly(mse)
        else:
            autoencoder = load_model("./pickle_files/models/autoencoder.keras")
            mse = anomaly_autoencoder.result_autocendoer(autoencoder, x_train)
            anomaly_autoencoder.Anomaly(mse)


if __name__ == "__main__":
    main()
