"""
DLNN con los SNP como entrada (comparación de arquitectura que pidió el editor de G3).

Misma entrada que la 1D-CNN de marcadores (SNP 0/1/2 estandarizados con las líneas de entrenamiento), mismos pliegues
y mismo protocolo anidado; solo cambia la arquitectura (DLNN en lugar de 1D-CNN). Se corre en los dos espacios:
    ESPACIO=v6 -> compacto (dlnn_v6, 60 trials)      ESPACIO=v7 -> amplio (dlnn_v3, 100 trials)
No toca ningún estudio, peso ni resultado existente: solo agrega el modelo "DLNN-SNP".

Uso (en /workspace/trigo_v6, dentro del contenedor):   bash lanzar_dlnn_snp.sh
Salidas por año y espacio:
    Opt_DLNN-SNP_<v6|v7>_<año>.db
    Resultados_<v6|v7>_<año>/unimodales/DLNN-SNP_metricas.csv  y  DLNN-SNP_predicciones.csv
"""
import importlib
import os
import sys

AÑO = os.environ.get("ANIO", "Y21")
ESPACIO = os.environ.get("ESPACIO", "v6")
assert ESPACIO in ("v6", "v7") and AÑO in ("Y21", "Y22")
os.environ["ANIO_V6" if ESPACIO == "v6" else "ANIO_V7"] = AÑO
ct = importlib.import_module(f"correr_todo_{ESPACIO}")

NOMBRE = "DLNN-SNP"
#          (nombre, entrada, tipo,  constructor,                               sc_x, pca_x, n_pca, depth)
ct.CFG[NOMBRE] = (NOMBRE, "SNP", "dlnn", "dlnn_v6" if ESPACIO == "v6" else "dlnn_v3", True, False, 1, 1)


def predicciones(y, X, part):
    """Predicciones por línea desde los modelos guardados (igual que predicciones_unimodales_v6/v7)."""
    import joblib
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from scipy.stats import pearsonr
    from sklearn.preprocessing import StandardScaler

    salida = f"{ct.RES}/unimodales/{NOMBRE}_predicciones.csv"
    carpeta = f"{ct.PESOS}/{NOMBRE}/completo"
    yv = y.set_index("GID")["BLUE_ton_ha"]
    fold_de = part.set_index("GID")["Fold_Prueba"]
    Xe = X["SNP"].set_index("GID")
    filas, comp = [], []
    for k in sorted(fold_de.unique()):
        pre = joblib.load(f"{carpeta}/fold_{k}_prepro.joblib")
        g_fit = [str(g) for g in pre["gid_fit"]]; g_ev = [str(g) for g in pre["gid_eval"]]
        assert set(g_ev) == set(fold_de.index[fold_de == k]), f"fold {k}: la prueba no coincide"
        Xt = Xe.loc[g_ev].to_numpy(dtype=float)
        if pre.get("scaler_X") is not None:
            Xt = pre["scaler_X"].transform(Xt)
        sy = StandardScaler().fit(yv.loc[g_fit].to_numpy().reshape(-1, 1))
        m = tf.keras.models.load_model(f"{carpeta}/fold_{k}.keras", compile=False)
        p = sy.inverse_transform(m.predict(Xt, verbose=0).reshape(-1, 1)).ravel()
        yr = yv.loc[g_ev].to_numpy()
        filas.append(pd.DataFrame({"GID": g_ev, "Fold_Prueba": k, "y_Real": yr, "y_Pred": p}))
        comp.append(pearsonr(yr, p)[0])
        tf.keras.backend.clear_session()
    met = pd.read_csv(f"{ct.RES}/unimodales/{NOMBRE}_metricas.csv").sort_values("fold")
    dif = np.max(np.abs(np.array(comp) - met["correlation"].to_numpy()))
    pd.concat(filas).to_csv(salida, index=False)
    print(f"{NOMBRE} {ESPACIO} {AÑO}: r = {np.mean(comp):.4f} ± {np.std(comp, ddof=1):.4f} | "
          f"dif. máx. con metricas.csv = {dif:.5f} [{'OK' if dif < 1e-3 else 'REVISAR'}]")


if __name__ == "__main__":
    y, X, part = ct.cargar()
    print(f"{NOMBRE} | espacio {ESPACIO} | año {AÑO} | {len(y)} líneas | {X['SNP'].shape[1] - 1} SNP | "
          f"{ct.N_UNI} trials | protocolo {ct.PROTOCOLO}")
    ct.unimodal(NOMBRE, y, X, part)          # búsqueda + examen final; el proceso se reinicia (código 3) tras cada tarea
    if not (ct.hecho(f"busqueda_{NOMBRE}") and ct.hecho(f"final_{NOMBRE}")):
        print("[!] Alguna tarea falló; revisa el log."); sys.exit(1)
    predicciones(y, X, part)
    ct.avisar(f"[DLNN-SNP {ESPACIO} {AÑO}] COMPLETO")
