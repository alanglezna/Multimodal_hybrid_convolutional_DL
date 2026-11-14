"""Número de parámetros de las redes individuales (revisor 1): reconstruye cada red con los hiperparámetros elegidos
por Optuna y cuenta sus parámetros. NO entrena ni escribe nada más que parametros_unimodales.csv. Tarda segundos.
Uso (en /workspace/trigo_v6):   python contar_parametros.py      (puede correrse mientras la GPU está ocupada)"""
import glob
import importlib
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # en CPU, para no estorbar a las corridas
import optuna
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

optuna.logging.set_verbosity(optuna.logging.WARNING)
filas = []
for esp in ("v6", "v7"):
    for año in ("Y21", "Y22"):
        os.environ["ANIO_V6" if esp == "v6" else "ANIO_V7"] = año
        ct = importlib.reload(importlib.import_module(f"correr_todo_{esp}"))
        opt = ct.opt
        y, X, part = ct.cargar()
        cfg = dict(ct.CFG)
        cfg["DLNN-SNP"] = ("DLNN-SNP", "SNP", "dlnn", "dlnn_v6" if esp == "v6" else "dlnn_v3", True, False, 1, 1)
        for nombre, (_, ent, tipo, fn, sc, pca, npca, depth) in cfg.items():
            bd = f"Opt_{nombre}_{esp}_{año}.db"
            if not os.path.exists(bd):
                continue
            try:
                st = optuna.load_study(study_name=f"{nombre}_{esp}_{año}", storage=f"sqlite:///{bd}")
                p = X[ent].shape[1] - 1
                forma = (p // depth, depth) if tipo == "cnn" else (p,)
                ent_k, sal_k = opt.SELECTOR_FUNCIONES[fn](optuna.trial.FixedTrial(st.best_trial.params))(forma, tipo)
                n = Model(ent_k, Dense(1)(sal_k)).count_params()
                filas.append({"estudio": f"{nombre}_{esp}_{año}", "entradas": p, "n_param": int(n)})
                print(f"{nombre:13s} {esp} {año}: {p:6d} entradas -> {n:9,d} parámetros")
            except Exception as e:
                print(f"[!] {nombre} {esp} {año}: {e}")
pd.DataFrame(filas).to_csv("parametros_unimodales.csv", index=False)
print(f"{len(filas)} redes -> parametros_unimodales.csv")
