"""
Modelo híbrido multimodal (fusión tardía de ramas preentrenadas y congeladas).

Flujo (función hibrido_v2):
1) Búsqueda Optuna de la cabeza de fusión con los pesos de fase 'interno'
   (ramas entrenadas con el 64% de las líneas); objetivo = MSE medio de validación (16%).
2) Examen final con los pesos de fase 'completo' (ramas entrenadas con el 80%):
   se reentrena la cabeza con épocas fijas (best_epoch_mean) y se evalúa en el 20% de prueba.
3) Importancia por permutación de cada modalidad sobre el conjunto de prueba.

Los pesos y el preprocesamiento de cada rama los genera
optuna_cnn1d_dlnn.entrenar_unimodal_para_hibrido:
    <carpeta_rama>/<fase>/fold_k.keras
    <carpeta_rama>/<fase>/fold_k_prepro.joblib   ({scaler_X, pca, gid_fit, gid_eval})
"""
import datetime
import gc
import os
import time

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # la GPU ya estaba inicializada por otro módulo

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import funciones

RAMAS = ("a", "snp", "dm")


# ======================================================================
# Constructor de la cabeza de fusión
# ======================================================================
def crear_fusion_dinamica_V3(trial):
    """Igual que V1, con la regla del protocolo v3: L2 1e-6-1e-1 (antes 1e-4-1e-2) y dropout 0.0-0.5 (antes 0.1-0.5)."""
    def build_fusion(combinado):
        z = layers.BatchNormalization(name="bn_fusion_in")(combinado)
        n_capas = trial.suggest_int("fusion_layers", 1, 3)
        l2 = trial.suggest_float("fusion_l2", 1e-6, 1e-1, log=True)
        for i in range(n_capas):
            unidades = trial.suggest_categorical(f"fusion_units_l{i}", [16, 32, 64, 128])
            z = layers.Dense(unidades, use_bias=False, kernel_regularizer=regularizers.l2(l2),
                             name=f"dense_fusion_{i}")(z)
            z = layers.BatchNormalization(name=f"bn_fusion_{i}")(z)
            z = layers.Activation("relu", name=f"act_fusion_{i}")(z)
            drop = trial.suggest_float(f"fusion_drop_l{i}", 0.0, 0.5)
            z = layers.Dropout(drop, name=f"drop_fusion_{i}")(z)
        return layers.Dense(1, activation="linear", name="Output_Layer_Fusion")(z)
    return build_fusion


def crear_fusion_dinamica_V6(trial):
    """Fusión pequeña (v6): BatchNorm opcional, 1-2 capas densas ({8,16,32} y {8,16}) con L2 y dropout -> salida lineal."""
    def build_fusion(combinado):
        usar_bn = trial.suggest_categorical("fusion_batchnorm", [False, True])
        z = layers.BatchNormalization(name="bn_fusion_in")(combinado) if usar_bn else combinado
        l2 = trial.suggest_float("fusion_l2", 1e-5, 1e-1, log=True)
        n_capas = trial.suggest_int("fusion_layers", 1, 2)
        opciones = {0: [8, 16, 32], 1: [8, 16]}
        for i in range(n_capas):
            z = layers.Dense(trial.suggest_categorical(f"fusion_units_l{i}", opciones[i]), use_bias=not usar_bn,
                             kernel_regularizer=regularizers.l2(l2), name=f"dense_fusion_{i}")(z)
            if usar_bn:
                z = layers.BatchNormalization(name=f"bn_fusion_{i}")(z)
            z = layers.Activation("relu", name=f"act_fusion_{i}")(z)
            z = layers.Dropout(trial.suggest_float(f"fusion_drop_l{i}", 0.0, 0.5), name=f"drop_fusion_{i}")(z)
        return layers.Dense(1, activation="linear", name="Output_Layer_Fusion")(z)
    return build_fusion


SELECTOR_FUNCIONES = {"fusion_V3": crear_fusion_dinamica_V3,     # fusión amplia (v7)
                      "fusion_V6": crear_fusion_dinamica_V6}     # fusión compacta (v6 e híbrido final)


def _aplicar_prepro(X, prepro):
    """Aplica el scaler y la PCA guardados con la rama (sin reajustar nada)."""
    if prepro.get("scaler_X") is not None:
        X = prepro["scaler_X"].transform(X)
    if prepro.get("pca") is not None:
        X = prepro["pca"].transform(X)
    return X


# ======================================================================
# Pipeline del híbrido
# ======================================================================
class Pipeline_Hibrido_V2:

    def __init__(self, modelo_fusion, learning_rate=0.001, batch_size=16, epochs=100,
                 seed=345322, verbose_model=0, cnn_a=False, cnn_snp=True, depth_snp=1,
                 proj_dim=None, extra_callbacks=None, paciencia=22, min_delta=0.0, paciencia_lr=10):
        self.build_fusion_func = modelo_fusion
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.verbose_model = verbose_model
        self.cnn_a = cnn_a
        self.cnn_snp = cnn_snp
        self.depth_snp = depth_snp
        self.proj_dim = proj_dim or None          # 0 o None = sin proyección
        self.extra_callbacks = extra_callbacks or []
        self.paciencia, self.min_delta, self.paciencia_lr = paciencia, min_delta, paciencia_lr

        self.scaler_y = StandardScaler()
        self.model = None
        self.history = None
        self.ramas_activas = []
        self._ent_eval, self._y_eval = None, None

    # ------------------------------------------------------------ utilidades
    def _formatear(self, rama, X):
        if rama == "snp" and self.cnn_snp:
            return X.reshape(-1, X.shape[1] // self.depth_snp, self.depth_snp)
        if rama == "a" and self.cnn_a:
            return X.reshape(-1, X.shape[1], 1)
        return X

    def build_and_compile(self, pesos):
        """Carga cada rama, la corta antes de su capa de salida, la congela
        (incluida BatchNormalization) y añade la proyección opcional antes de concatenar."""
        entradas, salidas = [], []
        for rama in self.ramas_activas:
            base = tf.keras.models.load_model(pesos[rama], compile=False)
            for capa in base.layers:
                capa._name = f"{capa.name}_{rama.upper()}"
            sub = Model(base.input, base.layers[-2].output, name=f"extractor_{rama}")
            sub.trainable = False

            z = sub.output
            if self.proj_dim:
                z = layers.Dense(self.proj_dim, use_bias=False, name=f"proj_{rama}")(z)
                z = layers.BatchNormalization(name=f"bn_proj_{rama}")(z)
                z = layers.Activation("relu", name=f"act_proj_{rama}")(z)
            print(f"[!] Rama {rama.upper()}: {sub.output.shape[-1]}D -> {z.shape[-1]}D | {pesos[rama]}")
            entradas.append(sub.input)
            salidas.append(z)

        if not salidas:
            raise ValueError("No se cargó ninguna rama.")
        combinado = layers.concatenate(salidas, name="Fusion_Latente") if len(salidas) > 1 else salidas[0]
        print(f"Neuronas en la fusión: {combinado.shape[-1]}")

        self.model = Model(entradas, self.build_fusion_func(combinado), name="Red_Multimodal")
        self.model.compile(optimizer=Adam(self.learning_rate), loss="mse", metrics=["mse", "mae"])

    # ------------------------------------------------------- entrenamiento
    def entrenar_y_evaluar(self, y_fit, y_eval, X_fit, X_eval, pesos, prepros, con_validacion):
        """
        con_validacion=True  -> búsqueda: X_eval es la validación (EarlyStopping sobre ella).
        con_validacion=False -> examen final: X_eval es la prueba (épocas fijas, sin validación).
        """
        tf.keras.backend.clear_session()
        funciones.set_seed(self.seed)

        self.scaler_y = StandardScaler().fit(y_fit.reshape(-1, 1))
        y_fit_sc = self.scaler_y.transform(y_fit.reshape(-1, 1)).ravel()
        y_eval_sc = self.scaler_y.transform(y_eval.reshape(-1, 1)).ravel()

        self.ramas_activas = [r for r in RAMAS if r in X_fit]
        ent_fit = [self._formatear(r, _aplicar_prepro(X_fit[r], prepros[r])) for r in self.ramas_activas]
        ent_eval = [self._formatear(r, _aplicar_prepro(X_eval[r], prepros[r])) for r in self.ramas_activas]

        self.build_and_compile(pesos)

        if con_validacion:
            callbacks = [EarlyStopping(monitor="val_mse", patience=self.paciencia, min_delta=self.min_delta,
                                       restore_best_weights=True),
                         ReduceLROnPlateau(monitor="val_mse", patience=self.paciencia_lr, factor=0.5, min_lr=1e-6)]
            val_data = (ent_eval, y_eval_sc)
        else:
            callbacks = [ReduceLROnPlateau(monitor="loss", patience=self.paciencia_lr, factor=0.5, min_lr=1e-6)]
            val_data = None

        self.history = self.model.fit(
            ent_fit, y_fit_sc, epochs=self.epochs, batch_size=self.batch_size,
            validation_data=val_data, verbose=self.verbose_model,
            callbacks=callbacks + self.extra_callbacks,
        )

        y_pred = self._predecir(ent_eval)
        mse = mean_squared_error(y_eval, y_pred)
        res = {
            "correlation": pearsonr(y_eval, y_pred)[0],
            "r2": r2_score(y_eval, y_pred),
            "rmse": np.sqrt(mse),
            "mae_real": mean_absolute_error(y_eval, y_pred),
            "bias": float(np.mean(y_pred - y_eval)),
            "n_param_total": int(self.model.count_params()),
            "n_param_entrenables": int(sum(np.prod(w.shape) for w in self.model.trainable_weights)),
        }
        if con_validacion:
            h = self.history.history["val_mse"]
            res["val_mse"] = float(np.min(h))
            res["ep_mejor"] = int(np.argmin(h) + 1)

        self._ent_eval, self._y_eval = ent_eval, y_eval
        return res, y_pred

    def _predecir(self, entradas):
        p = self.model.predict(entradas, verbose=0).reshape(-1, 1)
        return self.scaler_y.inverse_transform(p).ravel()

    def importancia_permutacion(self, y_pred_base, n_rep=5, seed=0):
        """Permuta cada modalidad completa (todas sus columnas juntas) en la prueba y mide la caída de r."""
        rng = np.random.default_rng(seed)
        r_base = pearsonr(self._y_eval, y_pred_base)[0]
        caida = {}
        for k, rama in enumerate(self.ramas_activas):
            valores = []
            for _ in range(n_rep):
                ent = list(self._ent_eval)
                ent[k] = ent[k][rng.permutation(len(self._y_eval))]
                valores.append(r_base - pearsonr(self._y_eval, self._predecir(ent))[0])
            caida[rama] = float(np.mean(valores))
        return caida


# ======================================================================
# Búsqueda + examen final
# ======================================================================
def hibrido_v2(
    y_data, df_maestro, X_ramas, carpetas_pesos,
    nombre_estudio, nombre_bd, funcion_dinamica="fusion_V6",
    n_trials=80, epo=500, SEED=345322,
    cnn_a=False, cnn_snp=True, dep=1,
    output_folder="Resultados_Hibrido_V2", numero_trial=None, protocolo="v6",
):
    """
    X_ramas:        {"a": X_A, "snp": X_snp, "dm": X_img}  DataFrames con GID, SIN preprocesar
                    (los mismos que se usaron en entrenar_unimodal_para_hibrido).
    carpetas_pesos: {"a": "Pesos/DLNN-AO", "snp": "Pesos/CNN-SNP", "dm": "Pesos/DLNN-DM6"}
    numero_trial:   None = mejor trial por validación; un número = evaluar ese trial.
    """
    funciones.set_seed(SEED)
    from optuna_1dcnn_dlnn import PROTOCOLOS
    P = PROTOCOLOS[protocolo]
    ramas = [r for r in RAMAS if r in X_ramas]

    # ---------- Alineación de GID ----------
    gids = y_data["GID"].astype(str).values
    for r in ramas:
        if not np.array_equal(X_ramas[r]["GID"].astype(str).values, gids):
            raise ValueError(f"La rama '{r}' no tiene los GID en el mismo orden que y_data.")
    pos = {g: i for i, g in enumerate(gids)}

    Y = y_data.drop(columns=["GID"]).values.ravel()
    X = {r: X_ramas[r].drop(columns=["GID"]).values for r in ramas}

    maestro = df_maestro[["GID", "Fold_Prueba"]].copy()
    maestro["GID"] = maestro["GID"].astype(str)
    mapeo = pd.DataFrame({"GID": gids}).merge(maestro, on="GID", how="left")["Fold_Prueba"].values
    folds = np.sort(np.unique(mapeo)).astype(int)

    # ---------- Pesos y preprocesamiento por fase ----------
    def cargar_fase(fase, fold):
        pesos, prepros = {}, {}
        for r in ramas:
            base = os.path.join(carpetas_pesos[r], fase)
            pesos[r] = os.path.join(base, f"fold_{fold}.keras")
            prepros[r] = joblib.load(os.path.join(base, f"fold_{fold}_prepro.joblib"))
        ref = prepros[ramas[0]]
        for r in ramas[1:]:
            if not (np.array_equal(prepros[r]["gid_fit"], ref["gid_fit"])
                    and np.array_equal(prepros[r]["gid_eval"], ref["gid_eval"])):
                raise ValueError(f"Fold {fold}, fase {fase}: las ramas no usaron el mismo split.")
        idx_fit = np.array([pos[g] for g in ref["gid_fit"]])
        idx_eval = np.array([pos[g] for g in ref["gid_eval"]])
        return pesos, prepros, idx_fit, idx_eval

    constructor = SELECTOR_FUNCIONES[funcion_dinamica]

    def nuevo_pipeline(fusion, lr, batch, epocas, proj):
        return Pipeline_Hibrido_V2(modelo_fusion=fusion, learning_rate=lr, batch_size=batch,
                                   epochs=epocas, seed=SEED, cnn_a=cnn_a, cnn_snp=cnn_snp,
                                   depth_snp=dep, proj_dim=proj, paciencia=P["paciencia"],
                                   min_delta=P["min_delta"], paciencia_lr=P["paciencia_lr"])

    # ---------- Búsqueda Optuna (solo validación interna) ----------
    def objective(trial):
        if protocolo == "v6":
            lr = trial.suggest_float("learning_rate", P["lr_min"], P["lr_max"], log=True)
            proj = trial.suggest_categorical("proj_dim", [0, 8, 16])
        else:
            lr = trial.suggest_float("learning_rate", P["lr_min"], 1e-2, log=True)          # protocolo v3
            proj = trial.suggest_categorical("proj_dim", [0, 16, 32])
        batch = trial.suggest_categorical("batch_size", [16, 32, 64])
        fusion = constructor(trial)

        mses, epocas = [], []
        for paso, fold in enumerate(folds, start=1):
            gc.collect()
            pesos, prepros, i_fit, i_val = cargar_fase("interno", fold)
            pipe = nuevo_pipeline(fusion, lr, batch, epo, proj)
            res, _ = pipe.entrenar_y_evaluar(
                Y[i_fit], Y[i_val],
                {r: X[r][i_fit] for r in ramas}, {r: X[r][i_val] for r in ramas},
                pesos, prepros, con_validacion=True,
            )
            mses.append(res["val_mse"])
            epocas.append(res["ep_mejor"])
            trial.report(float(np.mean(mses)), step=paso)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("best_epoch_mean", int(np.mean(epocas)))
        trial.set_user_attr("best_epoch_median", int(np.median(epocas)))
        trial.set_user_attr("epocas_por_fold", [int(e) for e in epocas])
        return float(np.mean(mses))

    inicio = time.time()
    storage_db = optuna.storages.RDBStorage(url=nombre_bd, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(
        direction="minimize", study_name=nombre_estudio, storage=storage_db, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=271167),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=15),
    )
    from optuna_1dcnn_dlnn import TRIALS_POR_PROCESO, guardia_memoria, trials_validos, rss_gb
    restantes = n_trials - trials_validos(study)          # los trials fallidos no cuentan
    if restantes > 0:
        lote = min(restantes, TRIALS_POR_PROCESO)
        print(f"[*] {nombre_estudio}: {restantes}/{n_trials} trials restantes; este proceso corre {lote} "
              f"(RSS inicial {rss_gb():.1f} GB)")
        study.optimize(objective, n_trials=lote, callbacks=[guardia_memoria])
        if trials_validos(study) < n_trials:
            print(f"[*] {nombre_estudio}: faltan trials; continúa en otro proceso")
            return None, None, study

    if numero_trial is None:
        best = study.best_trial
    else:
        best = study.trials[numero_trial]
        if best.state != optuna.trial.TrialState.COMPLETE:
            raise ValueError(f"El trial #{numero_trial} no está completo ({best.state.name}).")
    ep_final = best.user_attrs["best_epoch_median" if P["epocas"] == "mediana" else "best_epoch_mean"]
    print(f"\n[!] Trial evaluado #{best.number} | val_mse={best.value:.5f} | épocas={ep_final}")
    print(f"    Parámetros: {best.params}")

    # ---------- Examen final contra la prueba ----------
    fusion = constructor(best)
    resultados, filas = [], []
    for fold in folds:
        gc.collect()
        pesos, prepros, i_fit, i_te = cargar_fase("completo", fold)
        if not np.array_equal(np.sort(i_te), np.where(mapeo == fold)[0]):
            raise ValueError(f"Fold {fold}: la prueba guardada no coincide con Fold_Prueba.")

        pipe = nuevo_pipeline(fusion, best.params["learning_rate"], best.params["batch_size"],
                              ep_final, best.params["proj_dim"])
        res, y_pred = pipe.entrenar_y_evaluar(
            Y[i_fit], Y[i_te],
            {r: X[r][i_fit] for r in ramas}, {r: X[r][i_te] for r in ramas},
            pesos, prepros, con_validacion=False,
        )
        res["fold"] = fold
        for rama, dr in pipe.importancia_permutacion(y_pred, n_rep=5, seed=SEED + fold).items():
            res[f"dr_{rama}"] = dr
        resultados.append(res)
        filas.append(pd.DataFrame({"GID": gids[i_te], "Fold": fold, "y_Real": Y[i_te], "y_Pred": y_pred}))
        print(f"    [FOLD {fold}] Corr: {res['correlation']:.4f} | RMSE: {res['rmse']:.4f}")

    # ---------- Importancia relativa (Δr negativo = sin dependencia) ----------
    df_res = pd.DataFrame(resultados)
    cols = [c for c in df_res.columns if c.startswith("dr_")]
    positivos = df_res[cols].clip(lower=0)
    pct = 100 * positivos.div(positivos.sum(axis=1), axis=0)
    pct.columns = [c.replace("dr_", "pct_") for c in cols]
    df_res = pd.concat([df_res, pct], axis=1)

    print("\nImportancia relativa por permutación (media ± DE entre folds):")
    for c in pct.columns:
        print(f"  {c[4:].upper():4s}: {pct[c].mean():5.1f} ± {pct[c].std():.1f} %")
    print(f"Parámetros: {int(df_res['n_param_total'].iloc[0]):,} totales | "
          f"{int(df_res['n_param_entrenables'].iloc[0]):,} entrenables")

    df_pred = pd.concat(filas, ignore_index=True)
    os.makedirs(output_folder, exist_ok=True)
    df_res.to_csv(os.path.join(output_folder, f"{nombre_estudio}_metricas.csv"), index=False)
    df_pred.to_csv(os.path.join(output_folder, f"{nombre_estudio}_predicciones.csv"), index=False)

    print("\n" + "=" * 60)
    print(f" Duración: {datetime.timedelta(seconds=int(time.time() - inicio))}")
    print(f" Correlación: {df_res['correlation'].mean():.4f} ± {df_res['correlation'].std():.4f}")
    print(f" RMSE:        {df_res['rmse'].mean():.4f} ± {df_res['rmse'].std():.4f}")
    print("=" * 60)
    return df_res, df_pred, study
