"""
Pipeline de entrenamiento para los modelos unimodales (1D-CNN y DLNN).

ModelPipeline_A2
- run_on_fold(X_train, y_train, X_test, y_test)
      Divide el entrenamiento en 80:20 interno (random_state=24591), entrena con early stopping
      sobre la validación interna y evalúa en X_test. Lo usa la búsqueda de Optuna.
- run_on_fold_train_100(X_train, y_train, X_test, y_test)
      Entrena con TODO X_train durante un número fijo de épocas (sin validación ni early stopping)
      y evalúa en X_test. Lo usa el examen final y los pesos de las ramas del híbrido.

En ambos casos, la estandarización de y, la estandarización de X y la PCA (opcionales) se ajustan
solo con las líneas de entrenamiento y se aplican sin cambios a las demás.
Atributos que se guardan junto con los pesos: scaler_X y pca.
"""
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import funciones


class ModelPipeline_A2:

    def __init__(
        self,
        model_builder_func,
        model_filepath="best_model.keras",
        verbose_model=1,
        learning_rate=0.001,
        batch_size=16,
        epochs=100,
        seed=345322,
        save=False,
        depth=1,
        scale_X=False,
        pca_X=False,
        n_pca=18,
        model_type="cnn",
        extra_callbacks=None,
        paciencia=22,
        min_delta=0.0,
        paciencia_lr=10,
    ):
        if model_type not in ("cnn", "dlnn"):
            raise ValueError("model_type debe ser 'cnn' o 'dlnn'")
        self.model_builder_func = model_builder_func
        self.model_filepath = model_filepath
        self.verbose_model = verbose_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.save = save
        self.depth = depth
        self.scale_X = scale_X
        self.pca_X = pca_X
        self.n_pca = n_pca
        self.model_type = model_type
        self.extra_callbacks = extra_callbacks or []
        # Valores por defecto = protocolo _v2 (no cambian los resultados existentes)
        self.paciencia = paciencia
        self.min_delta = min_delta
        self.paciencia_lr = paciencia_lr

        self.scaler_y = StandardScaler()
        self.scaler_X = StandardScaler()
        self.pca = PCA(n_components=self.n_pca)

        self.model = None
        self.history = None
        self.X_train_net, self.y_train_scaled = None, None
        self.X_val_net, self.y_val_scaled = None, None

        funciones.set_seed(self.seed)

    # ------------------------------------------------------------ utilidades
    def build_and_compile(self):
        input_shape = self.X_train_net.shape[1:]
        base_entrada, base_salida = self.model_builder_func(input_shape, self.model_type)
        salida = Dense(1, activation="linear", name="Output_Layer")(base_salida)
        self.model = Model(inputs=base_entrada, outputs=salida)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss="mse", metrics=["mse", "mae"])

    def _preprocesar_X(self, X_fit, *otros):
        """Estandarización y PCA opcionales, ajustadas solo con X_fit."""
        if not self.scale_X:
            return (X_fit, *otros)
        print("[!]-Escala de X")
        self.scaler_X = StandardScaler().fit(X_fit)
        X_fit = self.scaler_X.transform(X_fit)
        otros = [self.scaler_X.transform(o) for o in otros]
        if self.pca_X:
            self.pca = PCA(n_components=self.n_pca).fit(X_fit)
            print(f"   [PCA] {self.pca.n_components_} componentes "
                  f"({100 * self.pca.explained_variance_ratio_.sum():.2f}% de varianza)")
            X_fit = self.pca.transform(X_fit)
            otros = [self.pca.transform(o) for o in otros]
        return (X_fit, *otros)

    def _formatear(self, X):
        """CNN: (muestras, posiciones, canales); DLNN: tabla."""
        if self.model_type == "cnn":
            return X.reshape(-1, X.shape[1] // self.depth, self.depth)
        return X

    def _metricas(self, y_real, y_pred):
        mse = mean_squared_error(y_real, y_pred)
        return {
            "r2": r2_score(y_real, y_pred),
            "rmse": np.sqrt(mse),
            "mae_real": mean_absolute_error(y_real, y_pred),
            "correlation": pearsonr(y_real, y_pred)[0],
            "bias": float(np.mean(y_pred - y_real)),
        }

    def _predecir(self, X_net):
        p = self.model.predict(X_net, verbose=0).reshape(-1, 1)
        return self.scaler_y.inverse_transform(p).ravel()

    # ------------------------------------------- búsqueda (con validación)
    def run_on_fold(self, X_train, y_train, X_test, y_test, grupos_train=None):
        idx_tr, idx_val = train_test_split(np.arange(len(X_train)), test_size=0.20,
                                           random_state=24591, shuffle=True)
        tf.keras.backend.clear_session()
        X_tr, y_tr = X_train.iloc[idx_tr].values, y_train.iloc[idx_tr].values.ravel()
        X_val, y_val = X_train.iloc[idx_val].values, y_train.iloc[idx_val].values.ravel()
        X_te, y_te = X_test.values, y_test.values.ravel()

        self.scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        self.y_train_scaled = self.scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
        self.y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()

        X_tr, X_val, X_te = self._preprocesar_X(X_tr, X_val, X_te)
        self.X_train_net, self.X_val_net = self._formatear(X_tr), self._formatear(X_val)
        X_te_net = self._formatear(X_te)

        print(f"X_train: {self.X_train_net.shape}, y_train: {self.y_train_scaled.shape}")
        self.build_and_compile()
        callbacks = [
            EarlyStopping(monitor="val_mse", patience=self.paciencia, min_delta=self.min_delta,
                          restore_best_weights=True, start_from_epoch=15),
            ReduceLROnPlateau(monitor="val_mse", patience=self.paciencia_lr, factor=0.5, min_lr=1e-6),
        ] + self.extra_callbacks
        self.history = self.model.fit(
            self.X_train_net, self.y_train_scaled, epochs=self.epochs, batch_size=self.batch_size,
            validation_data=(self.X_val_net, self.y_val_scaled),
            verbose=self.verbose_model, callbacks=callbacks,
        )

        y_pred = self._predecir(X_te_net)
        res = self._metricas(y_te, y_pred)
        res["ep_mejor"] = int(np.argmin(self.history.history["val_mse"]) + 1)
        res["epocas"] = len(self.history.history["loss"])
        return res, y_pred

    # ------------------------------- entrenamiento final (épocas fijas)
    def run_on_fold_train_100(self, X_train, y_train, X_test, y_test):
        tf.keras.backend.clear_session()
        y_train = np.asarray(y_train).ravel(); y_test = np.asarray(y_test).ravel()

        self.scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        self.y_train_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).ravel()

        X_train, X_test = self._preprocesar_X(np.asarray(X_train), np.asarray(X_test))
        self.X_train_net, X_te_net = self._formatear(X_train), self._formatear(X_test)

        print(f"X_train: {self.X_train_net.shape}, y_train: {self.y_train_scaled.shape}")
        self.build_and_compile()
        callbacks = [ReduceLROnPlateau(monitor="loss", patience=self.paciencia_lr, factor=0.5, min_lr=1e-6)] \
            + self.extra_callbacks
        self.history = self.model.fit(
            self.X_train_net, self.y_train_scaled, epochs=self.epochs, batch_size=self.batch_size,
            validation_data=None, verbose=self.verbose_model, callbacks=callbacks,
        )

        y_pred = self._predecir(X_te_net)
        res = self._metricas(y_test, y_pred)
        res["ep_mejor"] = self.epochs
        res["epocas"] = len(self.history.history["loss"])
        print(f"Partición finalizada -> R²: {res['r2']:.4f}, Corr: {res['correlation']:.4f}, "
              f"RMSE: {res['rmse']:.2f}")
        return res, y_pred
