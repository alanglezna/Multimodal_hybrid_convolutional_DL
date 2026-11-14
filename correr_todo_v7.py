"""
SCRIPT MAESTRO v7 (artículo): los mismos datos, pliegues, flujo y regla de selección que v6, pero con las
ARQUITECTURAS Y EL PROTOCOLO DE v3 (los del artículo enviado a G3):
  - constructores dlnn_v3 y cnn_v3 (BatchNorm fijo, 1-3 capas/bloques), fusión fusion_V3 (1-3 capas de 16-128);
  - protocolo "v3": lr 1e-5-5e-3, EarlyStopping paciencia 40 (mejora mínima 1e-4), ReduceLROnPlateau 15,
    épocas finales = mediana de los 5 pliegues; 100 trials por modelo y por híbrido, 150 para el MMDL (como v3).
  - imágenes: solo DM6-AUC (32 métricas), estandarizadas y sin PCA, igual que el RKHS de R2.
  - dos años: Y21 (2020-21) y Y22 (2021-22, B5IR). Estudios, pesos y resultados con sufijo _v7 (no toca v6).

Flujo por año:
  parte 1: DLNN-AO, DLNN-GO, CNN-SNP, CNN-SNP-OHE, DLNN-DM6AUC y DLNN-DM (comparativa) (búsqueda + examen final)
  parte 2: selección de la rama de marcadores por MENOR val MSE (CNN-SNP vs CNN-SNP-OHE), híbrido trimodal
           + 3 bimodales, y MMDL de Montesinos (AO + GO + DM6AUC).
Uso:
    nohup bash lanzar_v7.sh Y21 1 > /dev/null 2>&1 &      # luego:  bash lanzar_v7.sh Y21 2
Si se interrumpe, se vuelve a lanzar: Optuna continúa y las tareas hechas se saltan.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import mmdl_montesinos as mm          # primero: activa el crecimiento de memoria de la GPU
import optuna
import funciones
import optuna_1dcnn_dlnn as opt
import Pipeline_Multimodal_hibrido as ph

AÑO = os.environ.get("ANIO_V7", "Y21")
PROTOCOLO = "v3"
N_UNI, N_HIB, N_MMDL = 100, 100, 150
REINICIAR_POR_TAREA = True


def nombres(año):
    global SUF, RES, PESOS
    SUF, RES, PESOS = f"_v7_{año}", f"Resultados_v7_{año}", f"Pesos_v7_{año}"


nombres(AÑO)
D = "Datos_entrenamiento/"


# ---------------------------------------------------------------------- datos
def cargar():
    yf = pd.read_csv(f"{D}y_folds_{AÑO}.csv")
    yf["GID"] = yf["GID"].astype(str)
    orden = yf["GID"].values
    y = yf[["GID", "y"]].rename(columns={"y": "BLUE_ton_ha"})
    part = yf[["GID", "Fold_Prueba"]]

    def alinear(ruta):
        X = pd.read_csv(ruta); X["GID"] = X["GID"].astype(str)
        X = X.set_index("GID").loc[orden].reset_index()
        assert np.array_equal(X["GID"].values, orden) and not X.isna().any().any(), ruta
        return X

    snp = alinear(f"{D}SNP_{AÑO}.csv")
    M = snp.drop(columns="GID").to_numpy()
    ohe = np.stack([(M == 0), (M == 1), (M == 2)], axis=2).reshape(M.shape[0], -1).astype("int8")   # locus por locus
    cols = [f"{c}_{k}" for c in snp.columns[1:] for k in (0, 1, 2)]
    X = {
        "AO":      alinear(f"{D}AO_{AÑO}.csv"),
        "GO":      alinear(f"{D}GO_{AÑO}.csv"),
        "SNP":     snp,
        "SNP-OHE": pd.concat([snp[["GID"]], pd.DataFrame(ohe, columns=cols)], axis=1),
        "DM":      alinear(f"{D}IMG_DM_{AÑO}.csv"),
        "DM6":     alinear(f"{D}IMG_DM6_{AÑO}.csv"),
        "DM6AUC":  alinear(f"{D}IMG_DM6AUC_{AÑO}.csv"),
    }
    return y, X, part


# ------------------------------------------------- configuración de unimodales
# (nombre, entrada, tipo, constructor, sc_x, pca_x, n_pca, depth)
CONFIG = [
    ("DLNN-AO",     "AO",      "dlnn", "dlnn_v3", False, False, 1, 1),
    ("DLNN-GO",     "GO",      "dlnn", "dlnn_v3", False, False, 1, 1),
    ("CNN-SNP",     "SNP",     "cnn",  "cnn_v3",  True,  False, 1, 1),
    ("CNN-SNP-OHE", "SNP-OHE", "cnn",  "cnn_v3",  False, False, 1, 3),
    ("DLNN-DM6AUC", "DM6AUC",  "dlnn", "dlnn_v3", True,  False, 1, 1),
    ("DLNN-DM",     "DM",      "dlnn", "dlnn_v3", True,  False, 1, 1),     # comparativa: última fecha (no entra al híbrido)
]
CFG = {c[0]: c for c in CONFIG}
# Imágenes: solo DM6-AUC (la metodología descrita en el artículo); DLNN-DM y DLNN-DM6 quedan como individuales informativos
CANDIDATAS = {"a": ["DLNN-AO"], "snp": ["CNN-SNP", "CNN-SNP-OHE"], "dm": ["DLNN-DM6AUC"]}
ES_RAMA = {m for v in CANDIDATAS.values() for m in v}          # guardan pesos 'interno' y 'completo'


# ----------------------------------------------------------------- utilidades
def bd(nombre):
    return f"sqlite:///Opt_{nombre}{SUF}.db"

def hecho(tarea):
    return os.path.exists(os.path.join(RES, "hechos", tarea))

def marcar(tarea):
    os.makedirs(os.path.join(RES, "hechos"), exist_ok=True)
    open(os.path.join(RES, "hechos", tarea), "w").close()

def avisar(msg):
    print(msg)
    try:
        if os.environ.get("TELEGRAM_TOKEN"):
            funciones.enviar_telegram(msg)
    except Exception:
        pass

def tarea(nombre, fn):
    if hecho(nombre):
        print(f"[=] {nombre}: ya hecho, se salta"); return
    t0 = time.time()
    try:
        fn()
        marcar(nombre)
        avisar(f"[v7 {AÑO}] Terminó {nombre} ({(time.time() - t0) / 60:.0f} min)")
        if REINICIAR_POR_TAREA:
            print("[~] Reinicio del proceso para liberar memoria"); sys.stdout.flush()
            os._exit(3)
    except Exception as e:
        traceback.print_exc()
        avisar(f"[v7 {AÑO}] FALLÓ {nombre}: {e}  -> sigo con la siguiente tarea")

def continuar_en_otro_proceso(estudio, nombre_bd, n_trials):
    if opt.busqueda_incompleta(estudio, nombre_bd, n_trials):
        print(f"[~] {estudio}: reinicio del proceso para liberar memoria (RSS {opt.rss_gb():.1f} GB)")
        sys.stdout.flush(); os._exit(3)


# --------------------------------------------------------------- unimodales
def busqueda(nombre, y, X, part):
    _, ent, tipo, fn, sc, pca, npca, depth = CFG[nombre]
    estudio = nombre + SUF
    if tipo == "dlnn":
        opt.ejecutar_optimizacion_dlnn_global(X[ent], y, part, sc_x=sc, pca_x=pca, pca_n=npca, funcion_dinamica=fn,
                                              nombre_estudio=estudio, nombre_bd=bd(nombre), n_trials=N_UNI,
                                              protocolo=PROTOCOLO)
    else:
        opt.ejecutar_optimizacion_cnn_global(X[ent], y, part, sc_x=sc, pca_x=pca, depth=depth, nombre_estudio=estudio,
                                             nombre_bd=bd(nombre), funcion_dinamica=fn, n_trials=N_UNI,
                                             protocolo=PROTOCOLO)
    continuar_en_otro_proceso(estudio, bd(nombre), N_UNI)

def examen_final(nombre, y, X, part, fases=("completo",)):
    _, ent, tipo, fn, sc, pca, npca, depth = CFG[nombre]
    for fase in fases:
        df = opt.entrenar_unimodal_para_hibrido(
            X_data=X[ent], y_data=y, df_maestro=part, output_folder=f"{PESOS}/{nombre}",
            nombre_estudio=nombre + SUF, nombre_bd=bd(nombre), funcion_dinamica=fn,
            model_type=tipo, sc_x=sc, pca_x=pca, n_pca=npca, depth=depth, fase=fase, protocolo=PROTOCOLO)
        if fase == "completo":
            os.makedirs(f"{RES}/unimodales", exist_ok=True)
            df.to_csv(f"{RES}/unimodales/{nombre}_metricas.csv", index=False)

def unimodal(nombre, y, X, part):
    tarea(f"busqueda_{nombre}", lambda: busqueda(nombre, y, X, part))
    fases = ("interno", "completo") if nombre in ES_RAMA else ("completo",)
    tarea(f"final_{nombre}", lambda: examen_final(nombre, y, X, part, fases))


# ------------------------------------------------- selección de ramas (menor val MSE)
def seleccionar_ramas():
    """Para cada fuente elige el modelo candidato con MENOR val MSE (MSE medio de validación interna del mejor trial).
    Regla fijada antes de correr. Se guarda en ramas_elegidas.json (lo usa también R2 para el RKHS trimodal)."""
    eleccion, tabla = {}, []
    for rama, cands in CANDIDATAS.items():
        vals = {}
        for m in cands:
            st = optuna.load_study(study_name=m + SUF, storage=bd(m))
            vals[m] = float(st.best_value)
            tabla.append({"rama": rama, "modelo": m, "val_mse": vals[m], "trials": len(st.trials)})
        eleccion[rama] = min(vals, key=vals.get)
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(tabla).to_csv(f"{RES}/seleccion_ramas.csv", index=False)
    with open(f"{RES}/ramas_elegidas.json", "w") as fh:
        json.dump({**eleccion, "entrada": {r: CFG[m][1] for r, m in eleccion.items()}}, fh, indent=1)
    print("[*] Ramas elegidas por val MSE:", eleccion)
    print(pd.DataFrame(tabla).to_string(index=False))
    return eleccion


# ----------------------------------------------------------------- híbridos
def hibrido(ramas, eleccion, y, X, part):
    ent = {r: CFG[eleccion[r]][1] for r in ramas}
    nombre = "Hibrido_" + "_".join(ent[r] for r in ramas)
    X_r = {r: X[ent[r]] for r in ramas}
    pesos = {r: f"{PESOS}/{eleccion[r]}" for r in ramas}
    dep = CFG[eleccion["snp"]][7] if "snp" in ramas else 1

    def fn():
        ph.hibrido_v2(y_data=y, df_maestro=part, X_ramas=X_r, carpetas_pesos=pesos,
                      nombre_estudio=nombre + SUF, nombre_bd=bd(nombre), funcion_dinamica="fusion_V3",
                      n_trials=N_HIB, cnn_snp=("snp" in ramas), dep=dep, output_folder=f"{RES}/hibridos",
                      protocolo=PROTOCOLO)
        continuar_en_otro_proceso(nombre + SUF, bd(nombre), N_HIB)
    tarea(nombre, fn)

def montesinos(eleccion, y, X, part):
    dm = CFG[eleccion["dm"]][1]
    def fn():
        mm.mmdl_montesinos(y_data=y, df_maestro=part, X_ramas={"a": X["AO"], "go": X["GO"], "dm": X[dm]},
                           nombre_estudio="MMDL_Montesinos" + SUF, nombre_bd=bd("MMDL_Montesinos"),
                           n_trials=N_MMDL, n_pca=None, output_folder=f"{RES}/montesinos")
        continuar_en_otro_proceso("MMDL_Montesinos" + SUF, bd("MMDL_Montesinos"), N_MMDL)
    tarea("MMDL_Montesinos", fn)


# --------------------------------------------------------------------- flujo
def correr(parte=0):
    """parte=1: los 6 individuales | parte=2: selección de ramas, 4 híbridos y Montesinos | 0: todo en secuencia."""
    y, X, part = cargar()
    print(f"Año {AÑO}: {len(y)} líneas | BLUE medio = {y['BLUE_ton_ha'].mean():.3f} | protocolo {PROTOCOLO} | parte {parte}")
    for e, df in X.items():
        print(f"  {e:8s}: {df.shape[1] - 1} columnas")
    if parte in (0, 1):
        for nombre, *_ in CONFIG:
            unimodal(nombre, y, X, part)
    if parte in (0, 2):
        faltan = [m for m in ES_RAMA if not hecho(f"final_{m}")]
        if faltan:
            print(f"[!] Faltan modelos individuales: {faltan}. Corre primero la parte 1."); return
        eleccion = seleccionar_ramas()
        for ramas in (("a", "snp", "dm"), ("a", "snp"), ("a", "dm"), ("snp", "dm")):
            hibrido(ramas, eleccion, y, X, part)
        montesinos(eleccion, y, X, part)
    avisar(f"[v7 {AÑO}] PARTE {parte} COMPLETA")
