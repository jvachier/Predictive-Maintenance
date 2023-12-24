import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os.path


from sklearn.metrics import (
    precision_recall_fscore_support,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn import metrics, datasets

# from functools import reduce


import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam


class Predictions:
    "Add description"

    def __init__(self, df) -> None:
        self.data: pd.DataFrame = df

    def train_split(self) -> list:
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop(columns=["datetime", "failure"]).values,
            self.data["failure"].values,
            test_size=0.20,
            stratify=self.data["failure"].values,
            random_state=1,
        )
        return X_train, X_test, y_train, y_test

    def model_lr(self, X_train, y_train: list, X_test):
        pipe_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=1, solver="lbfgs", max_iter=10000),
        ).fit(X_train, y_train)
        y_predic_lr = pipe_lr.predict(X_test)
        y_predic_lr_proba = pipe_lr.predict_proba(X_test)
        return pipe_lr, y_predic_lr, y_predic_lr_proba

    def model_RF(self, X_train, y_train: list, X_test):
        clf_RFC = RandomForestClassifier(
            n_estimators=25,
            max_depth=10,
            max_features="sqrt",
            random_state=1,
            n_jobs=4,
        ).fit(X_train, y_train)
        y_pred_RFC = clf_RFC.predict(X_test)
        y_pred_RFC_proba = clf_RFC.predict_proba(X_test)
        return clf_RFC, y_pred_RFC, y_pred_RFC_proba

    def model_metrics(self, y_test: list, y_pred: list, name: str) -> None:
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:, 1])
        score = metrics.auc(fpr, tpr)
        print("AUC score " + str(name) + ":" + str(score))

    def roc_curve(self, model1, model2, X_test, y_test: list) -> None:
        plt.figure(1)
        models = [model1, model2]
        x_tests = [X_test, X_test]
        y_tests = [y_test, y_test]

        names = ["Logistic Rgression", "Random Forest"]

        for i, j in enumerate(models):
            ax = plt.gca()
            clf_disp = RocCurveDisplay.from_estimator(
                j, x_tests[i], y_tests[i], ax=ax, name=names[i], alpha=0.8
            )
        plt.show()

    def visualization_prediction(self, y_test: list, y_pred: list, name: str) -> None:
        plt.figure(2)
        plt.plot(y_test, "+", label="Real")
        plt.plot(y_pred, ".", label="Predicted")
        plt.ylabel("Target")
        plt.legend()
        plt.title(str(name))
        plt.show()

    def visualization_accuracy(self, model, name: str, X_train, y_train: list) -> None:
        plt.figure(3)
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=10,
            n_jobs=2,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.plot(
            train_sizes,
            train_mean,
            color="b",
            marker="o",
            markersize=5,
            label="Training accuracy",
        )

        plt.fill_between(
            train_sizes,
            train_mean + train_std,
            train_mean - train_std,
            alpha=0.5,
            color="b",
        )

        plt.plot(
            train_sizes,
            test_mean,
            color="g",
            marker="s",
            linestyle="--",
            markersize=5,
            label="Validation accuracy",
        )

        plt.fill_between(
            train_sizes,
            test_mean + test_std,
            test_mean - test_std,
            alpha=0.15,
            color="g",
        )

        plt.grid()
        plt.xlabel("Number of training")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.title(str(name))
        # plt.savefig("./figures/accuracy_" + str(name) + ".png")
        plt.show()


class Save_Load_models:
    def __init__(self) -> None:
        pass

    def save_model_sklearn(
        self, name: str, model, prediction: np.array, prediction_proba: np.array
    ) -> None:
        dbfile_model = open("./pickle_files/models/" + str(name), "ab")
        dbfile_prediction = open(
            "./pickle_files/models/" + str(name) + "_predictions", "ab"
        )
        dbfile_prediction_proba = open(
            "./pickle_files/models/" + str(name) + "_predictions_proba", "ab"
        )
        pickle.dump(model, dbfile_model)
        pickle.dump(prediction, dbfile_prediction)
        pickle.dump(prediction_proba, dbfile_prediction_proba)
        dbfile_model.close()
        dbfile_prediction.close()
        dbfile_prediction_proba.close()

    def load_model_sklearn(self, name: str):
        dbfile_model = open("./pickle_files/models/" + str(name), "rb")
        dbfile_prediction = open(
            "./pickle_files/models/" + str(name) + "_predictions", "rb"
        )
        dbfile_prediction_proba = open(
            "./pickle_files/models/" + str(name) + "_predictions_proba", "rb"
        )
        model_loaded = pickle.load(dbfile_model)
        predictions_loaded = pickle.load(dbfile_prediction)
        prediction_proba_loaded = pickle.load(dbfile_prediction_proba)
        dbfile_model.close()
        dbfile_prediction.close()
        dbfile_prediction_proba.close()
        return model_loaded, predictions_loaded, prediction_proba_loaded


class Anomaly_detection_isolationforest:
    def __init__(self, df: pd.DataFrame, feature_name: str):
        self.data: pd.DataFrame = df
        self.name: str = feature_name
        self.scaler_iso = None

    def isolationforest(self) -> None:
        data_index = self.data.set_index("datetime")
        self.scaler_iso = StandardScaler()
        np_scaled = self.scaler_iso.fit_transform(data_index.values.reshape(-1, 1))
        data = pd.DataFrame(np_scaled)
        model_time_series_t = IsolationForest(n_estimators=500, contamination=0.1)
        model_time_series_t.fit(data.values)
        self.data["anomaly"] = model_time_series_t.predict(data.values)

    def visulaization_isolationforest(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        a = self.data.loc[self.data["anomaly"] == -1, [self.name]]  # anomaly
        ax.plot(
            self.data.index,
            self.data[self.name],
            color="black",
            label="Normal",
        )
        ax.scatter(a.index, a[self.name], color="red", label="Anomaly")
        plt.legend()
        plt.ylabel(self.name, fontsize=13)
        plt.xlabel("Time", fontsize=13)
        # plt.savefig('anomaly_extruder_mass_temperature.png')
        plt.show()


class Anomaly_detection_autoencoder:
    def __init__(self, df: pd.DataFrame, feature_name: str):
        self.data: pd.DataFrame = df
        self.name: str = feature_name
        self.scaler_auto = None

    def data_to_feed_autoencoder(self) -> np.array:
        self.scaler_auto = StandardScaler()
        self.data[self.name] = self.scaler_auto.fit_transform(self.data[[self.name]])
        data = self.data.drop(columns=["datetime", "anomaly"], axis=1).values
        # data = self.data[self.name].values
        x_train = self._create_sequences(data, 50)
        return x_train

    def result_autocendoer(self, model, x_train: np.array) -> tf:
        # Calculate the reconstruction error for each data point
        reconstructions_deep = model.predict(x_train)

        mse = tf.reduce_mean(tf.square(x_train - reconstructions_deep), axis=[1, 2])
        return mse

    def AutoEncoder(self, x_train: np.array):
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
        encoded = Dense(128, activation="relu")(input_layer)
        encoded = Dense(64, activation="relu")(encoded)
        encoded = Dense(32, activation="relu")(encoded)

        decoded = Dense(64, activation="relu")(encoded)
        decoded = Dense(128, activation="relu")(decoded)
        decoded = Dense(x_train.shape[2], activation="relu")(decoded)

        # Compile and fit the model
        autoencoder_deep = Model(input_layer, decoded)
        autoencoder_deep.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )
        autoencoder_deep.fit(
            x_train, x_train, epochs=10, batch_size=128, validation_split=0.1
        )

        if os.path.isfile("./pickle_files/models/autoencoder.keras") is False:
            autoencoder_deep.save("./pickle_files/models/autoencoder.keras")
        return autoencoder_deep

    def Anomaly(self, mse: tf) -> None:
        anomaly_deep_scores = pd.Series(mse.numpy(), name="anomaly_scores")
        anomaly_deep_scores.index = self.data[49:].index
        anomaly_deep_scores = pd.Series(mse.numpy(), name="anomaly_scores")
        anomaly_deep_scores.index = self.data[49:].index

        threshold_deep = anomaly_deep_scores.quantile(0.98)
        anomalous_deep = anomaly_deep_scores > threshold_deep
        binary_labels_deep = anomalous_deep.astype(int)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            binary_labels_deep,
            anomalous_deep,
            # average='binary'
        )

        plt.figure(figsize=(16, 8))
        plt.plot(
            self.data["datetime"],
            self.scaler_auto.inverse_transform(self.data[[self.name]]),
            "k",
        )
        plt.plot(
            self.data["datetime"][49:][anomalous_deep],
            self.scaler_auto.inverse_transform(
                self.data[[self.name]][49:][anomalous_deep]
            ),
            "ro",
        )
        plt.title("Anomaly Detection")
        plt.xlabel("Time")
        plt.ylabel(self.name)
        # plt.savefig('anomaly_extruder_mass_pressure_autoencoder_higher_complexity.png')
        plt.show()

    def _create_sequences(self, values: list, time_steps: int) -> np.array:
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)
