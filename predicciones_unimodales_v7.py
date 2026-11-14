"""
Predicciones por línea de los modelos individuales de DL (v6/v7), SIN reentrenar.

El examen final guardó los modelos (fold_k.keras) y su preprocesamiento (fold_k_prepro.joblib), pero no las
predicciones por línea, que hacen falta para la r dentro de ensayo, el top 20 % y las pruebas pareadas.
Este script carga cada modelo de la fase 'completo', predice las líneas de prueba de su pliegue y regresa a t/ha
con el mismo StandardScaler de y (ajustado con las líneas de entrenamiento del pliegue).
Comprueba que la r por pliegue coincide con la de <modelo>_metricas.csv.

Uso (en /workspace/trigo_v6, dentro del contenedor):
    ANIO_V7=Y21 python predicciones_unimodales_v7.py
    ANIO_V7=Y22 python predicciones_unimodales_v7.py
Salida: Resultados_v7_<año>/unimodales/<modelo>_predicciones.csv  (GID, Fold_Prueba, y_Real, y_Pred)
"""
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

import correr_todo_v7 as ct

y, X, part = ct.cargar()
yv = y.set_index("GID")["BLUE_ton_ha"]
fold_de = part.set_index("GID")["Fold_Prueba"]

for nombre, ent, tipo, fn, sc, pca, npca, depth in ct.CONFIG:
    carpeta = f"{ct.PESOS}/{nombre}/completo"
    if not os.path.isdir(carpeta):
        print(f"[!] {nombre}: no existe {carpeta}, se salta"); continue
    Xe = X[ent].set_index("GID")
    filas, comp = [], []
    for k in sorted(fold_de.unique()):
        pre = joblib.load(f"{carpeta}/fold_{k}_prepro.joblib")
        g_fit = [str(g) for g in pre["gid_fit"]]; g_ev = [str(g) for g in pre["gid_eval"]]
        assert set(g_ev) == set(fold_de.index[fold_de == k]), f"{nombre} fold {k}: la prueba no coincide"
        Xt = Xe.loc[g_ev].to_numpy(dtype=float)
        if pre.get("scaler_X") is not None:
            Xt = pre["scaler_X"].transform(Xt)
        if pre.get("pca") is not None:
            Xt = pre["pca"].transform(Xt)
        if tipo == "cnn":
            Xt = Xt.reshape(-1, Xt.shape[1] // depth, depth)
        sy = StandardScaler().fit(yv.loc[g_fit].to_numpy().reshape(-1, 1))
        m = tf.keras.models.load_model(f"{carpeta}/fold_{k}.keras", compile=False)
        p = sy.inverse_transform(m.predict(Xt, verbose=0).reshape(-1, 1)).ravel()
        yr = yv.loc[g_ev].to_numpy()
        filas.append(pd.DataFrame({"GID": g_ev, "Fold_Prueba": k, "y_Real": yr, "y_Pred": p}))
        comp.append(pearsonr(yr, p)[0])
        tf.keras.backend.clear_session()
    met = pd.read_csv(f"{ct.RES}/unimodales/{nombre}_metricas.csv").sort_values("fold")
    dif = np.max(np.abs(np.array(comp) - met["correlation"].to_numpy()))
    estado = "OK" if dif < 1e-3 else "REVISAR"
    pd.concat(filas).to_csv(f"{ct.RES}/unimodales/{nombre}_predicciones.csv", index=False)
    print(f"{nombre:13s} r = {np.mean(comp):.4f} | dif. máx. con metricas.csv = {dif:.5f} [{estado}]")
