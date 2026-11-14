"""
Modelos unimodales: 1D-CNN y DLNN con búsqueda de hiperparámetros en Optuna.

Contenido
- crear_constructor_dlnn_v6 / _cnn_v6   espacio de búsqueda COMPACTO (correr_todo_v6.py)
- crear_constructor_dlnn_v3 / _cnn_v3   espacio de búsqueda AMPLIO (correr_todo_v7.py)
- ejecutar_optimizacion_cnn_global   búsqueda de Optuna para la 1D-CNN (solo búsqueda)
- ejecutar_optimizacion_dlnn_global  búsqueda de Optuna para la DLNN (solo búsqueda)
- entrenar_unimodal_para_hibrido     examen final (fase 'completo') y pesos de las ramas ('interno' y 'completo')

Protocolo: cada trial se entrena y evalúa en las 5 particiones externas, con una división interna
fija 80:20 del entrenamiento externo (random_state=24591). El objetivo es el MSE medio de validación
interna. El examen final se hace con entrenar_unimodal_para_hibrido.
"""
import gc
import os

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers

import funciones
import Pipeline_1dcnn_dlnn as pipeline_cnn_dlnn


# ======================================================================
# CONSTRUCTORES
# ======================================================================
def crear_constructor_cnn_v3(trial):
    """Igual que 'cnn', con L2 de las capas densas en 1e-6-1e-1 (antes 1e-5-1e-3)."""
    def build_cnn(input_shape, nam):
        inp = layers.Input(shape=input_shape, name=nam)
        x = inp
        alpha_val = trial.suggest_float("leaky_alpha", 0.01, 0.2)
        for i in range(trial.suggest_int("n_conv_layers", 1, 3)):
            filtros = trial.suggest_categorical(f"conv{i}_filtros", [8, 16, 32, 64])
            kernel = trial.suggest_categorical(f"conv{i}_kernel", [3, 5, 7, 9])
            x = layers.Conv1D(filtros, kernel, padding="same", strides=2, use_bias=True)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=alpha_val)(x)
            x = layers.SpatialDropout1D(trial.suggest_float(f"conv{i}_drop", 0.1, 0.3))(x)
        x = layers.AveragePooling1D(pool_size=trial.suggest_categorical("pool_size", [2, 4, 6]))(x)
        x = layers.Flatten()(x)
        for i in range(trial.suggest_int("n_dense_layers", 1, 2)):
            unidades = trial.suggest_categorical(f"dense{i}_units", [16, 32, 64])
            l1_val = trial.suggest_float(f"dense{i}_l1", 1e-8, 1e-4, log=True)
            l2_val = trial.suggest_float(f"dense{i}_l2", 1e-6, 1e-1, log=True)
            x = layers.Dense(unidades, use_bias=True,
                             kernel_regularizer=regularizers.l1_l2(l1=l1_val, l2=l2_val))(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=alpha_val)(x)
            x = layers.Dropout(trial.suggest_float(f"dense{i}_drop", 0.2, 0.4))(x)
        return inp, x
    return build_cnn


def crear_constructor_dlnn_v3(trial):
    """Igual que 'dlnn', con L2 en 1e-6-1e-1 (antes 1e-5-1e-2) y dropout en 0.0-0.5 (antes 0.1-0.5)."""
    def build_dlnn(input_shape, nam):
        inp = layers.Input(shape=input_shape, name=nam)
        x = inp
        activacion = trial.suggest_categorical("activacion", ["relu", "leaky_relu"])
        alpha = trial.suggest_float("leaky_alpha", 0.01, 0.2) if activacion == "leaky_relu" else None
        for i in range(trial.suggest_int("n_layers", 1, 3)):
            x = layers.Dense(trial.suggest_categorical(f"units_l{i}", [16, 32, 64, 128, 256, 512]),
                             use_bias=False,
                             kernel_regularizer=regularizers.l2(
                                 trial.suggest_float(f"l2_l{i}", 1e-6, 1e-1, log=True)))(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=alpha)(x) if alpha else layers.Activation("relu")(x)
            x = layers.Dropout(trial.suggest_float(f"drop_l{i}", 0.0, 0.5))(x)
        return inp, x
    return build_dlnn


def crear_constructor_dlnn_v6(trial):
    """DLNN pequeña (v6): 1-2 capas densas con neuronas independientes por capa, ReLU, L2 y dropout;
    BatchNorm opcional (lo decide Optuna)."""
    def build_dlnn(input_shape, nam):
        inp = layers.Input(shape=input_shape, name=nam)
        x = inp
        usar_bn = trial.suggest_categorical("batchnorm", [False, True])
        n_capas = trial.suggest_int("n_layers", 1, 2)
        opciones = {0: [16, 32, 64], 1: [8, 16, 32]}
        for i in range(n_capas):
            x = layers.Dense(trial.suggest_categorical(f"units_l{i}", opciones[i]), use_bias=not usar_bn,
                             kernel_regularizer=regularizers.l2(trial.suggest_float(f"l2_l{i}", 1e-5, 1e-1, log=True)))(x)
            if usar_bn:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(trial.suggest_float(f"drop_l{i}", 0.0, 0.5))(x)
        return inp, x
    return build_dlnn


def crear_constructor_cnn_v6(trial):
    """1D-CNN pequeña (v6): 1-2 bloques Conv1D (stride 2, ReLU), pooling promedio, aplanado y 1-2 capas densas
    pequeñas con L2 y dropout; BatchNorm opcional (lo decide Optuna)."""
    def build_cnn(input_shape, nam):
        inp = layers.Input(shape=input_shape, name=nam)
        x = inp
        usar_bn = trial.suggest_categorical("batchnorm", [False, True])
        kernel = trial.suggest_categorical("conv_kernel", [3, 5, 7])
        for i in range(trial.suggest_int("n_conv_layers", 1, 2)):
            x = layers.Conv1D(trial.suggest_categorical(f"conv{i}_filtros", [8, 16, 32]), kernel,
                              padding="same", strides=2, use_bias=not usar_bn)(x)
            if usar_bn:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.AveragePooling1D(pool_size=trial.suggest_categorical("pool_size", [4, 8, 16]), padding="same")(x)
        x = layers.Flatten()(x)
        drop = trial.suggest_float("drop", 0.0, 0.5)
        l2 = trial.suggest_float("l2", 1e-5, 1e-1, log=True)
        opciones = {0: [8, 16, 32], 1: [8, 16]}
        x = layers.Dropout(drop)(x)
        for i in range(trial.suggest_int("n_dense_layers", 1, 2)):
            x = layers.Dense(trial.suggest_categorical(f"dense{i}_units", opciones[i]), use_bias=not usar_bn,
                             kernel_regularizer=regularizers.l2(l2))(x)
            if usar_bn:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(drop)(x)
        return inp, x
    return build_cnn


SELECTOR_FUNCIONES = {
    "cnn_v3": crear_constructor_cnn_v3,
    "dlnn_v3": crear_constructor_dlnn_v3,
    "cnn_v6": crear_constructor_cnn_v6,
    "dlnn_v6": crear_constructor_dlnn_v6,
}

# ---------------------------------------------------------------- control de memoria
# La memoria del proceso crece trial a trial (TensorFlow/Optuna). Cada búsqueda corre a lo más
# TRIALS_POR_PROCESO trials por proceso y se detiene si la RSS pasa LIMITE_RSS_GB; el script
# maestro entonces termina el proceso (código 3) y lanzar_v3.sh lo reabre con memoria limpia.
TRIALS_POR_PROCESO = 10
LIMITE_RSS_GB = 7.0


def rss_gb():
    try:
        with open("/proc/self/status") as fh:
            for linea in fh:
                if linea.startswith("VmRSS:"):
                    return int(linea.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


def guardia_memoria(study, trial):
    r = rss_gb()
    if r > LIMITE_RSS_GB:
        print(f"[!] RSS = {r:.1f} GB > {LIMITE_RSS_GB} GB: se detiene el estudio en este proceso")
        study.stop()


def trials_validos(study):
    return sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
               for t in study.trials)


def busqueda_incompleta(nombre_estudio, nombre_bd, n_trials):
    """True si al estudio aún le faltan trials (para que el maestro reinicie el proceso)."""
    st = optuna.load_study(study_name=nombre_estudio, storage=nombre_bd)
    return trials_validos(st) < n_trials


# Protocolos de entrenamiento: "v3" = espacio amplio (correr_todo_v7.py); "v6" = espacio compacto (correr_todo_v6.py).
PROTOCOLOS = {
    "v3": dict(lr_min=1e-5, lr_max=5e-3, paciencia=40, min_delta=1e-4, paciencia_lr=15, epocas="mediana"),
    # v6: redes pequeñas (BatchNorm opcional), 60 trials, más paciencia para que la red pueda mejorar un poco más
    "v6": dict(lr_min=1e-4, lr_max=5e-3, paciencia=80, min_delta=1e-4, paciencia_lr=30, epocas="mediana"),
}


# ======================================================================
# BÚSQUEDA DE HIPERPARÁMETROS
# ======================================================================
def _mapeo_folds(X_data, y_data, df_maestro):
    X = X_data.copy(); y = y_data.copy()
    X["GID"] = X["GID"].astype(str); y["GID"] = y["GID"].astype(str)
    if not np.array_equal(X["GID"].values, y["GID"].values):
        raise ValueError("X_data y y_data no tienen los GID en el mismo orden.")
    maestro = df_maestro[["GID", "Fold_Prueba"]].copy()
    maestro["GID"] = maestro["GID"].astype(str)
    mapeo = X[["GID"]].merge(maestro, on="GID", how="left")["Fold_Prueba"]
    if mapeo.isna().any():
        raise ValueError("Existen GIDs en X_data sin Fold_Prueba en df_maestro.")
    return X.drop(columns=["GID"]), y.drop(columns=["GID"]), mapeo.values


def _optimizar(X_data, y_data, df_maestro, model_type, funcion_dinamica,
               sc_x, pca_x, n_pca, depth, nombre_estudio, nombre_bd, n_trials, seed, epochs,
               protocolo="v6"):
    """Búsqueda de Optuna común a CNN y DLNN. Solo busca; el examen final es aparte."""
    P = PROTOCOLOS[protocolo]
    X_k, Y_k, mapeo = _mapeo_folds(X_data, y_data, df_maestro)
    folds = np.sort(np.unique(mapeo)).astype(int)
    funciones.set_seed(seed)
    constructor = SELECTOR_FUNCIONES[funcion_dinamica]

    def objective(trial):
        lr = trial.suggest_float("learning_rate", P["lr_min"], P["lr_max"], log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        builder = constructor(trial)
        mses, epocas = [], []
        for paso, fold in enumerate(folds):
            K.clear_session(); tf.keras.backend.clear_session(); gc.collect()
            idx = np.where(mapeo != fold)[0]
            X_tr, y_tr = X_k.iloc[idx], Y_k.iloc[idx]
            pipeline = pipeline_cnn_dlnn.ModelPipeline_A2(
                model_builder_func=builder, verbose_model=0, learning_rate=lr,
                batch_size=batch_size, seed=seed, epochs=epochs, save=False, depth=depth,
                scale_X=sc_x, pca_X=pca_x, n_pca=n_pca, model_type=model_type,
                paciencia=P["paciencia"], min_delta=P["min_delta"], paciencia_lr=P["paciencia_lr"])
            # run_on_fold hace la división interna 80:20 (random_state=24591);
            # el "test" que recibe no se usa para nada en la búsqueda.
            pipeline.run_on_fold(X_tr, y_tr, X_tr.iloc[:2], y_tr.iloc[:2])
            h = pipeline.history.history["val_mse"]
            mses.append(float(np.min(h)))
            epocas.append(int(np.argmin(h) + 1))
            del pipeline; K.clear_session(); gc.collect()
            trial.report(float(np.mean(mses)), step=paso)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("best_epoch_mean", int(np.mean(epocas)))
        trial.set_user_attr("best_epoch_median", int(np.median(epocas)))
        trial.set_user_attr("epocas_por_fold", [int(e) for e in epocas])
        return float(np.mean(mses))

    storage = optuna.storages.RDBStorage(url=nombre_bd, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(
        direction="minimize", study_name=nombre_estudio, storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=271167),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=15),
    )
    restantes = n_trials - trials_validos(study)          # los trials fallidos no cuentan
    if restantes > 0:
        lote = min(restantes, TRIALS_POR_PROCESO)
        print(f"[*] {nombre_estudio}: {restantes}/{n_trials} trials restantes; este proceso corre {lote} "
              f"(RSS inicial {rss_gb():.1f} GB)")
        study.optimize(objective, n_trials=lote, callbacks=[guardia_memoria])
        if trials_validos(study) < n_trials:
            print(f"[*] {nombre_estudio}: faltan {n_trials - trials_validos(study)} trials; continúa en otro proceso")
            return study
    else:
        print(f"[*] {nombre_estudio}: estudio completo ({len(study.trials)} trials)")

    best = study.best_trial
    print(f"[!] Mejor trial #{best.number} | val_mse={best.value:.4f} | "
          f"épocas={best.user_attrs.get('best_epoch_mean')}")
    return study


def ejecutar_optimizacion_cnn_global(X_data, y_data, df_maestro, sc_x=False, pca_x=False, depth=1,
                                     nombre_estudio="Estudio_CNN", nombre_bd="sqlite:///Opt_CNN.db",
                                     funcion_dinamica="cnn", n_trials=100, seed=345322, epochs=500,
                                     n_pca=18, protocolo="v6"):
    return _optimizar(X_data, y_data, df_maestro, "cnn", funcion_dinamica, sc_x, pca_x, n_pca, depth,
                      nombre_estudio, nombre_bd, n_trials, seed, epochs, protocolo)


def ejecutar_optimizacion_dlnn_global(X_data, y_data, df_maestro, sc_x=False, pca_x=False, pca_n=18,
                                      funcion_dinamica="dlnn", nombre_estudio="Estudio_DLNN",
                                      nombre_bd="sqlite:///Opt_DLNN.db", n_trials=100, seed=345322,
                                      epochs=500, protocolo="v6"):
    return _optimizar(X_data, y_data, df_maestro, "dlnn", funcion_dinamica, sc_x, pca_x, pca_n, 1,
                      nombre_estudio, nombre_bd, n_trials, seed, epochs, protocolo)


# ======================================================================
# EXAMEN FINAL Y PESOS PARA EL HÍBRIDO (fases 'interno' y 'completo')
# ======================================================================
def entrenar_unimodal_para_hibrido(
    X_data, y_data, df_maestro, output_folder,
    nombre_estudio, nombre_bd, funcion_dinamica, model_type,
    sc_x=False, pca_x=False, n_pca=18, depth=1,
    fase="interno", numero_trial=None, seed=345322,
    val_size=0.20, seed_split=24591, protocolo="v6", epocas_por_fold=False,
):
    """
    fase="interno":  entrena con el 64% (train interno) y evalúa en el 16% de validación.
                     Pesos para la búsqueda de Optuna del híbrido.
    fase="completo": entrena con el 80% (train externo) y evalúa en el 20% de test.
                     Examen final del modelo unimodal y pesos para el examen final del híbrido.
    Usa épocas fijas (best_epoch_mean del mejor trial) y sin EarlyStopping.
    """
    assert fase in ("interno", "completo")
    funciones.set_seed(seed)

    X = X_data.copy(); y = y_data.copy()
    X["GID"] = X["GID"].astype(str); y["GID"] = y["GID"].astype(str)
    X_k, Y_k, mapeo = _mapeo_folds(X, y, df_maestro)
    gids = X["GID"].values

    study = optuna.load_study(study_name=nombre_estudio, storage=nombre_bd)
    trial = study.trials[numero_trial] if numero_trial is not None else study.best_trial
    params = trial.params
    P = PROTOCOLOS[protocolo]
    por_fold = None
    if epocas_por_fold:                      # análisis de sensibilidad: cada fold con sus propias épocas óptimas
        por_fold = trial.user_attrs["epocas_por_fold"]
        epocas = int(np.median(por_fold))
    else:
        epocas = trial.user_attrs["best_epoch_median" if P["epocas"] == "mediana" else "best_epoch_mean"]
    constructor = SELECTOR_FUNCIONES[funcion_dinamica]
    print(f"[*] {nombre_estudio} | Trial #{trial.number} | fase={fase} | épocas={por_fold or epocas}")

    carpeta = os.path.join(output_folder, fase)
    os.makedirs(carpeta, exist_ok=True)

    resultados = []
    for k, fold_val in enumerate(np.sort(np.unique(mapeo)).astype(int)):
        K.clear_session(); tf.keras.backend.clear_session(); gc.collect()
        if por_fold is not None:
            epocas = int(por_fold[k])        # mismo orden de folds que en la búsqueda

        idx_outer_tr = np.where(mapeo != fold_val)[0]
        idx_outer_te = np.where(mapeo == fold_val)[0]

        if fase == "interno":
            # MISMA división interna que la búsqueda de Optuna y que el híbrido
            pos_tr, pos_val = train_test_split(
                np.arange(len(idx_outer_tr)), test_size=val_size, random_state=seed_split
            )
            idx_fit, idx_eval = idx_outer_tr[pos_tr], idx_outer_tr[pos_val]
        else:
            idx_fit, idx_eval = idx_outer_tr, idx_outer_te

        pipeline = pipeline_cnn_dlnn.ModelPipeline_A2(
            model_builder_func=constructor(trial), verbose_model=0,
            learning_rate=params["learning_rate"], batch_size=params["batch_size"],
            seed=seed, epochs=epocas, depth=depth,
            scale_X=sc_x, pca_X=pca_x, n_pca=n_pca, model_type=model_type,
            paciencia_lr=P["paciencia_lr"],
        )
        res, _ = pipeline.run_on_fold_train_100(
            X_k.iloc[idx_fit].values, Y_k.iloc[idx_fit].values,
            X_k.iloc[idx_eval].values, Y_k.iloc[idx_eval].values,
        )

        pipeline.model.save(os.path.join(carpeta, f"fold_{fold_val}.keras"))
        joblib.dump(
            {
                "scaler_X": pipeline.scaler_X if sc_x else None,
                "pca": pipeline.pca if (sc_x and pca_x) else None,
                "gid_fit": gids[idx_fit],
                "gid_eval": gids[idx_eval],
            },
            os.path.join(carpeta, f"fold_{fold_val}_prepro.joblib"),
        )
        res["fold"] = fold_val
        resultados.append(res)

    df = pd.DataFrame(resultados)
    print(f"\n[{fase}] Correlación: {df['correlation'].mean():.4f} ± {df['correlation'].std():.4f}")
    return df
