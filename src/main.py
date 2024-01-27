import os.path

from keras.models import load_model

import modules.data_preparation as data_preparation
import modules.models as models
import modules.loading as loading

from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--prediction", action="store_true")

    args = parser.parse_args()

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

    DATA = data_preparation.Data_Preparation(
        df_telemetry,
        df_maintenance,
        df_errors,
        df_failures,
    )
    LOAD_DATA = data_preparation.Load_Save()

    if os.path.isfile("./pickle_files/data_preparation/data_set") is False:
        DATA.date_to_time()
        data_set = DATA.merge_df()
        data_set_prepared = DATA.df_prepared(data_set)
        LOAD_DATA.save_dataframe(data_set_prepared)
    else:
        data_set_prepared = LOAD_DATA.load_dataframe()

    if args.prediction:
        print("Models prediction\n")
        model = models.Predictions(data_set_prepared)

        save_model = models.Save_Load_models()

        X_train, X_test, y_train, y_test = model.train_split()

        if os.path.isfile("./pickle_files/models/lr") is False:
            clf_RFC = model.model_RF(25, 10, "sqrt", 4)
            pipe_lr = model.model_lr("lbfgs", 10000)

            clf_RFC_fit = model.optimize_model_hyper_RF(clf_RFC, X_train, y_train)

            pipe_lr_fit = model.fit_model(pipe_lr, X_train, y_train)

            y_predic_lr = pipe_lr_fit.predict(X_test)
            y_predic_lr_proba = pipe_lr_fit.predict_proba(X_test)

            y_pred_RFC = clf_RFC_fit.predict(X_test)
            y_pred_RFC_proba = clf_RFC_fit.predict_proba(X_test)

            save_model.save_model_sklearn(
                "lr",
                pipe_lr_fit,
                pipe_lr_fit.predict(X_test),
                pipe_lr_fit.predict_proba(X_test),
            )
            save_model.save_model_sklearn(
                "rf",
                clf_RFC_fit,
                clf_RFC_fit.predict(X_test),
                clf_RFC_fit.predict_proba(X_test),
            )
        else:
            pipe_lr_fit, y_predic_lr, y_predic_lr_proba = save_model.load_model_sklearn(
                "lr"
            )
            clf_RFC_fit, y_pred_RFC, y_pred_RFC_proba = save_model.load_model_sklearn(
                "rf"
            )

        model.model_metrics(y_test, y_predic_lr_proba, "Logictic Regression")
        model.model_metrics(y_test, y_pred_RFC_proba, "Random Forest")

        model.roc_curve(pipe_lr_fit, clf_RFC_fit, X_test, y_test)
        model.visualization_test(y_test, y_pred_RFC, "RF Test")

        model.visualization_accuracy(pipe_lr, "Logistic Regression", X_train, y_train)
        model.visualization_accuracy(clf_RFC, "Random Forest", X_train, y_train)

        #     pipe_lr, y_predic_lr, y_predic_lr_proba = model.model_lr(
        #         X_train, y_train, X_test
        #     )
        #     clf_RFC, y_pred_RFC, y_pred_RFC_proba = model.model_RF(
        #         X_train, y_train, X_test
        #     )
        #     save_model.save_model_sklearn("lr", pipe_lr, y_predic_lr, y_predic_lr_proba)
        #     save_model.save_model_sklearn("rf", clf_RFC, y_pred_RFC, y_pred_RFC_proba)
        # else:
        #     pipe_lr, y_predic_lr, y_predic_lr_proba = save_model.load_model_sklearn(
        #         "lr"
        #     )
        #     clf_RFC, y_pred_RFC, y_pred_RFC_proba = save_model.load_model_sklearn("rf")

        # model.model_metrics(y_test, y_predic_lr_proba, "Logictic Regression")
        # model.model_metrics(y_test, y_pred_RFC_proba, "Random Forest")

        # model.roc_curve(pipe_lr, clf_RFC, X_test, y_test)
        # model.visualization_prediction(y_test, y_pred_RFC, "RF")

        # model.visualization_accuracy(pipe_lr, "Logistic Regression", X_train, y_train)
        # model.visualization_accuracy(clf_RFC, "Random Forest", X_train, y_train)
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
            autoencoder = anomaly_autoencoder.AutoEncoder(x_train)
            mse = anomaly_autoencoder.result_autocendoer(autoencoder, x_train)
            anomaly_autoencoder.Anomaly(mse)
        else:
            autoencoder = load_model("./pickle_files/models/autoencoder.keras")
            mse = anomaly_autoencoder.result_autocendoer(autoencoder, x_train)
            anomaly_autoencoder.Anomaly(mse)


if __name__ == "__main__":
    main()
