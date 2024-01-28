import os.path
from argparse import ArgumentParser
from keras.models import load_model

from src.modules import data_preparation
from src.modules import models
from src.modules import loading


def main() -> None:
    """
    Main function including:
        - Loading data (loading.py)
        - Data Preparation (data_preparation.py)
        - Models & Visualization (models.py)
    """
    parser = ArgumentParser()
    parser.add_argument("--prediction", action="store_true")

    args = parser.parse_args()

    print("Loading data\n")

    load = loading.Loading_files()

    if os.path.isfile("./pickle_files/loading/telemetry") is False:
        (
            df_telemetry,
            df_machines,
            df_failures,
            df_errors,
            df_maintenance,
        ) = load.load_save_df()
    else:
        (
            df_telemetry,
            df_machines,
            df_failures,
            df_errors,
            df_maintenance,
        ) = load.load_db_file()

    data = data_preparation.Data_Preparation(
        df_telemetry,
        df_maintenance,
        df_errors,
        df_failures,
    )
    load_data = data_preparation.Load_Save()

    if os.path.isfile("./pickle_files/data_preparation/data_set") is False:
        data.date_to_time()
        data_set = data.merge_df()
        data_set_prepared = data.df_prepared(data_set)
        load_data.save_dataframe(data_set_prepared)
    else:
        data_set_prepared = load_data.load_dataframe()

    if args.prediction:
        print("Models prediction\n")
        model = models.Predictions(data_set_prepared)

        save_model = models.Save_Load_models()

        x_train, x_test, y_train, y_test = model.train_split()
        clf_rfc = model.model_rf(25, 10, "sqrt", 4)
        pipe_lr = model.model_lr("lbfgs", 10000)

        if os.path.isfile("./pickle_files/models/lr") is False:
            clf_rfc_fit = model.optimize_model_hyper_rf(clf_rfc, x_train, y_train)
            pipe_lr_fit = model.fit_model(pipe_lr, x_train, y_train)

            save_model.save_model_sklearn(
                "lr",
                pipe_lr_fit,
                pipe_lr_fit.predict(x_test),
                pipe_lr_fit.predict_proba(x_test),
            )
            save_model.save_model_sklearn(
                "rf",
                clf_rfc_fit,
                clf_rfc_fit.predict(x_test),
                clf_rfc_fit.predict_proba(x_test),
            )
        else:
            pipe_lr_fit, y_predic_lr, y_predic_lr_proba = save_model.load_model_sklearn(
                "lr"
            )
            clf_rfc_fit, y_pred_rfc, y_pred_rfc_proba = save_model.load_model_sklearn(
                "rf"
            )

        model.model_metrics(y_test, y_predic_lr_proba, "Logictic Regression")
        model.model_metrics(y_test, y_pred_rfc_proba, "Random Forest")

        model.roc_curve(pipe_lr_fit, clf_rfc_fit, x_test, y_test)
        model.visualization_prediction(y_test, y_pred_rfc, "RF Test")

        model.visualization_accuracy(pipe_lr, "Logistic Regression", x_train, y_train)
        model.visualization_accuracy(clf_rfc, "Random Forest", x_train, y_train)
    else:
        print("\n")
        print("Models Anomaly Detection\n")

        anomaly_isolation = models.Anomaly_detection_isolationforest(
            df_telemetry, "pressure", 1
        )
        anomaly_autoencoder = models.Anomaly_detection_autoencoder(
            df_telemetry, "pressure", 1, 5
        )

        anomaly_isolation.isolationforest()
        anomaly_isolation.visulaization_isolationforest()
        x_train = anomaly_autoencoder.data_to_feed_autoencoder()
        if os.path.isfile("./pickle_files/models/autoencoder.keras") is False:
            autoencoder = anomaly_autoencoder.autoencoder(x_train)
            mse = anomaly_autoencoder.result_autocendoer(autoencoder, x_train)
            anomaly_autoencoder.anomaly(mse)
        else:
            autoencoder = load_model("./pickle_files/models/autoencoder.keras")
            mse = anomaly_autoencoder.result_autocendoer(autoencoder, x_train)
            anomaly_autoencoder.anomaly(mse)


if __name__ == "__main__":
    main()
