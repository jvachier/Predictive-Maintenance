import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os.path

from dataclasses import dataclass
from typing import Tuple

from sklearn.metrics import (
    precision_recall_fscore_support,
    RocCurveDisplay,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn import metrics


import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam


@dataclass(slots=True)
class Predictions:
    data: pd.DataFrame

    def train_split(self) -> Tuple[np.array, np.array, list, list]:
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop(columns=["datetime", "failure"]).values,
            self.data["failure"].values,
            test_size=0.20,
            stratify=self.data["failure"].values,
            random_state=1,
        )
        return X_train, X_test, y_train, y_test

    def model_lr(
        self, X_train: np.array, y_train: list, X_test: np.array
    ) -> Tuple[object, np.array, np.array]:
        pipe_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=1, solver="lbfgs", max_iter=10000),
        ).fit(X_train, y_train)
        y_predic_lr = pipe_lr.predict(X_test)
        y_predic_lr_proba = pipe_lr.predict_proba(X_test)
        return pipe_lr, y_predic_lr, y_predic_lr_proba

    def model_RF(
        self, X_train: np.array, y_train: list, X_test: np.array
    ) -> Tuple[object, np.array, np.array]:
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

    def roc_curve(
        self, model1: object, model2: object, X_test: np.array, y_test: list
    ) -> None:
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
        plt.savefig("./figures/roc_curves.png")
        plt.show()

    def visualization_prediction(self, y_test: list, y_pred: list, name: str) -> None:
        plt.figure(2)
        plt.plot(y_test, "+", label="Real")
        plt.plot(y_pred, ".", label="Predicted")
        plt.ylabel("Target")
        plt.legend()
        plt.title(str(name))
        if os.path.isfile("./figures/predictions.png") is False:
            plt.savefig("./figures/predictions.png")
        plt.show()

    def visualization_accuracy(
        self, model: object, name: str, X_train, y_train: list
    ) -> None:
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
        if os.path.isfile("./figures/accuracy_" + str(name) + ".png") is False:
            plt.savefig("./figures/accuracy_" + str(name) + ".png")
        plt.show()


@dataclass(slots=True)
class Save_Load_models:
    def save_model_sklearn(
        self, name: str, model: object, prediction: np.array, prediction_proba: np.array
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


@dataclass(slots=True)
class Anomaly_detection_isolationforest:
    data: pd.DataFrame
    name: str
    machine_name: int
    scaler_iso: object = None

    def __post_init__(self):
        self.data = self.data.query("machineID == @self.machine_name")

    def isolationforest(self) -> None:
        data_index = self.data.set_index("datetime")
        self.scaler_iso = StandardScaler()
        np_scaled = self.scaler_iso.fit_transform(data_index.values)
        data = pd.DataFrame(np_scaled)
        model_time_series_t = IsolationForest(n_estimators=500, contamination=0.01)
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
        if os.path.isfile("./figures/anomaly_isolation_forest.png") is False:
            plt.savefig("./figures/anomaly_isolation_forest.png")
        plt.show()


@dataclass(slots=True)
class Anomaly_detection_autoencoder:
    data: pd.DataFrame
    name: str
    machine_name: int
    time_step: int
    scaler_auto: object = None

    def __post_init__(self):
        self.data = self.data.query("machineID == @self.machine_name")

    def data_to_feed_autoencoder(self) -> np.array:
        self.scaler_auto = StandardScaler()
        self.data[self.name] = self.scaler_auto.fit_transform(self.data[[self.name]])
        data = self.data.drop(columns=["datetime"], axis=1).values
        x_train = self._create_sequences(data)
        return x_train

    def result_autocendoer(self, model: object, x_train: np.array) -> tf:
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
            optimizer=Adam(learning_rate=0.00001), loss="mse", metrics=["accuracy"]
        )
        autoencoder_deep.fit(
            x_train, x_train, epochs=500, batch_size=128, validation_split=0.1
        )

        if os.path.isfile("./pickle_files/models/autoencoder.keras") is False:
            autoencoder_deep.save("./pickle_files/models/autoencoder.keras")
        return autoencoder_deep

    def Anomaly(self, mse: tf) -> None:
        anomaly_deep_scores = pd.Series(mse.numpy(), name="anomaly_scores")
        anomaly_deep_scores.index = self.data[(self.time_step - 1) :].index
        anomaly_deep_scores = pd.Series(mse.numpy(), name="anomaly_scores")
        anomaly_deep_scores.index = self.data[(self.time_step - 1) :].index

        threshold_deep = anomaly_deep_scores.quantile(0.95)
        anomalous_deep = anomaly_deep_scores > threshold_deep
        binary_labels_deep = anomalous_deep.astype(int)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            binary_labels_deep,
            anomalous_deep,
        )

        plt.figure(figsize=(16, 8))
        plt.plot(
            self.data.index,
            self.scaler_auto.inverse_transform(self.data[[self.name]]),
            "k",
        )
        plt.plot(
            self.data.index[(self.time_step - 1) :][anomalous_deep],
            self.scaler_auto.inverse_transform(
                self.data[[self.name]][(self.time_step - 1) :][anomalous_deep]
            ),
            "ro",
        )
        plt.title("Anomaly Detection")
        plt.xlabel("Time")
        plt.ylabel(self.name)
        if os.path.isfile("./figures/anomaly_autoencoder.png") is False:
            plt.savefig("./figures/anomaly_autoencoder.png")
        plt.show()

    def _create_sequences(self, values: list) -> np.array:
        output = []
        for i in range(len(values) - self.time_step + 1):
            output.append(values[i : (i + self.time_step)])
        return np.stack(output)
