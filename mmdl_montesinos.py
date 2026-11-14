"""
Línea base multimodal adaptada de Montesinos-López et al. (2023, G3, jkad045).

Qué conserva del modelo original
- Una sub-red MLP por modalidad: capas densas con L2, BatchNorm antes de ReLU y dropout.
- Las dos primeras capas con N1 neuronas y después la mitad de la anterior (N1, N1, N1/2, N1/4, ...),
  con conexiones residuales.
- Fusión tardía: se concatenan las salidas de las sub-redes y una neurona lineal con L2 predice.
- Entrenamiento CONJUNTO de extremo a extremo, desde inicialización aleatoria (sin preentrenar).
- Adam con decaimiento exponencial del learning rate y búsqueda bayesiana de hiperparámetros.
- Para la parte genómica, una raíz de G (en el original, el factor de Cholesky; aquí, GO).

- Espacio de búsqueda de su Tabla 1: hasta 6 capas por rama, hasta 1024 neuronas (genómica) y 64 (NDVI),
  L2 y dropout compartidos entre ramas, lr en [1e-5, 1e-2], batch fijo de 32 y 150 iteraciones.

Qué se adapta a este conjunto de datos (diferencias a declarar en el artículo)
- Modalidades: AO (pedigrí), GO (genómica) y DM (multiespectral). No hay rama de año (una sola temporada).
  AO usa los mismos límites que la rama genómica del original.
- Optimización con Optuna-TPE en lugar de bayes_opt.
- Mismo protocolo de validación que el híbrido: búsqueda 64:16 dentro del train externo (mismo split,
  random_state=24591) y examen final entrenando con el 80% durante las épocas fijas del mejor trial.
  El original detenía con la pérdida de ENTRENAMIENTO (patience optimizada, máx. 128 épocas) y usaba
  un 10% de validación solo para el objetivo de la búsqueda; aquí se usa early stopping sobre val_mse
  para que ambos modelos compartan exactamente el mismo protocolo.
- El artículo no da la fórmula del decaimiento del learning rate ni detalla los atajos residuales;
  aquí: lr_época = lr0 * decay^época y atajo desde la segunda capa.
- DM se estandariza y se reduce con PCA dentro de cada partición (el original no describe escalado del NDVI).

Como el val_mse se calcula con la misma partición 64:16 y la misma estandarización de y,
es directamente comparable con el val_mse de la búsqueda del híbrido (mejor trial: 0.7068).
"""
import gc
import os
import time
import datetime

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

import funciones

RAMAS = ("a", "go", "dm")

# Para correr varios notebooks en la misma GPU: que TensorFlow no reserve toda la memoria.
# Solo surte efecto si se ejecuta antes de que TensorFlow use la GPU en ese kernel.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass


# ======================================================================
# ARQUITECTURA
# ======================================================================
def rama_mlp_residual(x, nombre, n_capas, n1, l2, drop):
    """Sub-red de una modalidad (Montesinos-López et al. 2023):
    - las dos primeras capas tienen N1 neuronas; a partir de la tercera, la mitad de la anterior;
    - Dense -> BatchNorm -> ReLU -> Dropout, con L2 en cada capa oculta;
    - atajo residual desde la segunda capa (identidad si el ancho no cambia, proyección lineal si cambia)."""
    for l in range(n_capas):
        unidades = n1 if l < 2 else max(n1 // (2 ** (l - 1)), 2)
        h = layers.Dense(unidades, kernel_regularizer=regularizers.l2(l2), name=f"{nombre}_dense{l}")(x)
        h = layers.BatchNormalization(name=f"{nombre}_bn{l}")(h)
        h = layers.Activation("relu", name=f"{nombre}_relu{l}")(h)
        h = layers.Dropout(drop, name=f"{nombre}_drop{l}")(h)
        if l > 0:
            if x.shape[-1] == unidades:
                atajo = x
            else:
                atajo = layers.Dense(unidades, use_bias=False, name=f"{nombre}_skip{l}")(x)
            h = layers.Add(name=f"{nombre}_add{l}")([h, atajo])
        x = h
    return x


def construir_mmdl(dims, hp):
    entradas, salidas = [], []
    for r in RAMAS:
        inp = layers.Input(shape=(dims[r],), name=f"in_{r}")
        # L2 y dropout compartidos entre modalidades, como en el original (un solo λ y un solo DO)
        z = rama_mlp_residual(inp, r, hp[f"{r}_capas"], hp[f"{r}_n1"], hp["l2"], hp["dropout"])
        entradas.append(inp)
        salidas.append(z)

    fusion = layers.concatenate(salidas, name="fusion_tardia")
    salida = layers.Dense(1, activation="linear",
                          kernel_regularizer=regularizers.l2(hp["l2"]), name="salida")(fusion)

    modelo = Model(entradas, salida, name="MMDL_Montesinos_adaptado")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(hp["learning_rate"]), loss="mse", metrics=["mse"])
    return modelo


# Espacio de búsqueda según la Tabla 1 de Montesinos-López et al. (2023), mapeando
# genómica -> AO y GO (hasta 6 capas, hasta 1024 neuronas) y NDVI -> DM (hasta 6 capas, hasta 64).
LIMITES_RAMA = {"a": (6, 1024), "go": (6, 1024), "dm": (6, 64)}


def sugerir_hp(trial):
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": 32,                                           # fijo en el original
        "decay": trial.suggest_float("decay", 0.95, 1.0),           # lr_época = lr0 * decay^época
        "l2": trial.suggest_float("l2", 1e-8, 1e-2, log=True),      # λ compartido
        "dropout": trial.suggest_float("dropout", 1e-4, 0.5),       # DO compartido
    }
    for r, (max_capas, max_n) in LIMITES_RAMA.items():
        hp[f"{r}_capas"] = trial.suggest_int(f"{r}_capas", 1, max_capas)
        hp[f"{r}_n1"] = trial.suggest_int(f"{r}_n1", 8, max_n, log=True)
    return hp


# ======================================================================
# PREPROCESAMIENTO Y ENTRENAMIENTO
# ======================================================================
def preparar(X, idx_fit, idx_eval, n_pca):
    """Escalado y PCA de DM ajustados SOLO con los individuos de entrenamiento."""
    fit, ev = {}, {}
    for r in RAMAS:
        a, b = X[r][idx_fit], X[r][idx_eval]
        if r == "dm":
            sc = StandardScaler().fit(a)
            a, b = sc.transform(a), sc.transform(b)
            if n_pca and n_pca < a.shape[1]:                    # v6: n_pca=None -> sin PCA (igual que las ramas)
                pca = PCA(n_components=n_pca).fit(a)
                a, b = pca.transform(a), pca.transform(b)
        fit[f"in_{r}"], ev[f"in_{r}"] = a, b
    return fit, ev


def entrenar(hp, X_fit, y_fit, X_eval, y_eval, epocas, con_validacion, seed):
    tf.keras.backend.clear_session()
    funciones.set_seed(seed)

    sy = StandardScaler().fit(y_fit.reshape(-1, 1))
    yf = sy.transform(y_fit.reshape(-1, 1)).ravel()
    ye = sy.transform(y_eval.reshape(-1, 1)).ravel()

    dims = {r: X_fit[f"in_{r}"].shape[1] for r in RAMAS}
    modelo = construir_mmdl(dims, hp)

    lr0, dec = hp["learning_rate"], hp["decay"]
    callbacks = [LearningRateScheduler(lambda e: lr0 * dec ** e)]
    if con_validacion:
        callbacks.append(EarlyStopping(monitor="val_mse", patience=22,
                                       restore_best_weights=True, start_from_epoch=15))

    historia = modelo.fit(
        X_fit, yf,
        validation_data=(X_eval, ye) if con_validacion else None,
        epochs=epocas, batch_size=hp["batch_size"], verbose=0, callbacks=callbacks,
    )
    pred = sy.inverse_transform(modelo.predict(X_eval, verbose=0).reshape(-1, 1)).ravel()
    return modelo, historia, pred


# ======================================================================
# BÚSQUEDA + EXAMEN FINAL
# ======================================================================
def mmdl_montesinos(
    y_data, df_maestro, X_ramas,
    nombre_estudio="MMDL_Montesinos_fiel", nombre_bd="sqlite:///Opt_MMDL_Montesinos.db",
    n_trials=150, epo=500, n_pca=18, SEED=345322, val_size=0.20, seed_split=24591,
    output_folder="Resultados_MMDL_Montesinos",
    seed_sampler=271167, solo_busqueda=False,
):
    """
    X_ramas: {"a": X_A, "go": X_GO, "dm": X_img}  DataFrames con GID, SIN preprocesar
             (X_img con sus 32 columnas; la PCA a n_pca se ajusta dentro de cada partición).

    En paralelo (varios notebooks con el mismo estudio y la misma base de datos):
      - n_trials es el TOTAL del estudio, sumando todos los procesos.
      - Usa un seed_sampler distinto en cada notebook para no repetir trials.
      - solo_busqueda=True en los notebooks auxiliares: buscan y terminan sin examen final.
    """
    funciones.set_seed(SEED)

    gids = y_data["GID"].astype(str).values
    for r in RAMAS:
        if not np.array_equal(X_ramas[r]["GID"].astype(str).values, gids):
            raise ValueError(f"La rama '{r}' no tiene los GID en el mismo orden que y_data.")

    Y = y_data.drop(columns=["GID"]).values.ravel().astype(float)
    X = {r: X_ramas[r].drop(columns=["GID"]).values.astype(float) for r in RAMAS}

    maestro = df_maestro[["GID", "Fold_Prueba"]].copy()
    maestro["GID"] = maestro["GID"].astype(str)
    mapeo = pd.DataFrame({"GID": gids}).merge(maestro, on="GID", how="left")["Fold_Prueba"].values
    if pd.isna(mapeo).any():
        raise ValueError("Hay GIDs sin Fold_Prueba.")
    folds = np.sort(np.unique(mapeo)).astype(int)

    def split_interno(fold):
        idx_outer_tr = np.where(mapeo != fold)[0]
        pos_fit, pos_val = train_test_split(np.arange(len(idx_outer_tr)),
                                            test_size=val_size, random_state=seed_split)
        return idx_outer_tr[pos_fit], idx_outer_tr[pos_val]   # mismo 64:16 que el híbrido

    # ---------- Búsqueda Optuna (sin tocar el test) ----------
    def objective(trial):
        hp = sugerir_hp(trial)
        mses, epocas = [], []
        for paso, fold in enumerate(folds, start=1):
            gc.collect()
            i_fit, i_val = split_interno(fold)
            Xf, Xv = preparar(X, i_fit, i_val, n_pca)
            _, hist, _ = entrenar(hp, Xf, Y[i_fit], Xv, Y[i_val], epo, True, SEED)
            h = hist.history["val_mse"]
            mses.append(float(np.min(h)))
            epocas.append(int(np.argmin(h) + 1))
            trial.report(float(np.mean(mses)), step=paso)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("best_epoch_mean", int(np.mean(epocas)))
        return float(np.mean(mses))

    inicio = time.time()
    storage = optuna.storages.RDBStorage(url=nombre_bd, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(
        direction="minimize", study_name=nombre_estudio, storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed_sampler),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=15),
    )
    from optuna_1dcnn_dlnn import TRIALS_POR_PROCESO, guardia_memoria, trials_validos, rss_gb
    restantes = n_trials - trials_validos(study)
    if restantes > 0:
        lote = min(restantes, TRIALS_POR_PROCESO)
        print(f"[*] {nombre_estudio}: {restantes}/{n_trials} trials restantes; este proceso corre {lote} "
              f"(RSS inicial {rss_gb():.1f} GB)")
        study.optimize(objective, n_trials=lote, callbacks=[guardia_memoria])
        if trials_validos(study) < n_trials:
            print(f"[*] {nombre_estudio}: faltan trials; continúa en otro proceso")
            return None, None, study

    if solo_busqueda:
        print("[*] Búsqueda terminada en este proceso (solo_busqueda=True). Sin examen final.")
        return None, None, study

    completos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completos) < len(study.trials):
        corriendo = sum(t.state == optuna.trial.TrialState.RUNNING for t in study.trials)
        if corriendo:
            print(f"[!] Aviso: {corriendo} trial(s) siguen corriendo en otro proceso. "
                  "Espera a que terminen y vuelve a llamar la función para el examen final.")

    best = study.best_trial
    hp = dict(best.params)
    hp.setdefault("batch_size", 32)
    ep_final = best.user_attrs["best_epoch_mean"]
    print(f"\n[!] Mejor trial #{best.number} | val_mse={best.value:.5f} | épocas={ep_final}")
    print(f"    Parámetros: {hp}")

    # ---------- Examen final: entrena con el 80%, evalúa en el 20% ----------
    resultados, filas = [], []
    for fold in folds:
        gc.collect()
        i_tr = np.where(mapeo != fold)[0]
        i_te = np.where(mapeo == fold)[0]
        Xtr, Xte = preparar(X, i_tr, i_te, n_pca)
        modelo, _, y_pred = entrenar(hp, Xtr, Y[i_tr], Xte, Y[i_te], ep_final, False, SEED)

        mse = mean_squared_error(Y[i_te], y_pred)
        res = {
            "fold": fold,
            "correlation": pearsonr(Y[i_te], y_pred)[0],
            "r2": r2_score(Y[i_te], y_pred),
            "rmse": np.sqrt(mse),
            "mae_real": mean_absolute_error(Y[i_te], y_pred),
            "bias": float(np.mean(y_pred - Y[i_te])),
            "n_parametros": modelo.count_params(),
        }
        resultados.append(res)
        filas.append(pd.DataFrame({"GID": gids[i_te], "Fold": fold, "y_Real": Y[i_te], "y_Pred": y_pred}))
        print(f"    [FOLD {fold}] Corr: {res['correlation']:.4f} | RMSE: {res['rmse']:.4f}")

    df_res = pd.DataFrame(resultados)
    df_pred = pd.concat(filas, ignore_index=True)
    os.makedirs(output_folder, exist_ok=True)
    df_res.to_csv(os.path.join(output_folder, f"{nombre_estudio}_metricas.csv"), index=False)
    df_pred.to_csv(os.path.join(output_folder, f"{nombre_estudio}_predicciones.csv"), index=False)

    print("\n" + "=" * 60)
    print(f" Duración:    {datetime.timedelta(seconds=int(time.time() - inicio))}")
    print(f" Correlación: {df_res['correlation'].mean():.4f} ± {df_res['correlation'].std():.4f}")
    print(f" RMSE:        {df_res['rmse'].mean():.4f} ± {df_res['rmse'].std():.4f}")
    print(f" Parámetros:  {int(df_res['n_parametros'].iloc[0]):,}")
    print("=" * 60)
    return df_res, df_pred, study
