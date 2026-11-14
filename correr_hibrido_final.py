"""
Híbrido FINAL: ramas ya entrenadas, sin reentrenar ninguna red individual.
  - pedigrí:     DLNN-AO de v6 (red pequeña)            -> Pesos_v6_<año>/DLNN-AO
  - marcadores:  la CNN que eligió v7 (arquitectura v3) -> Pesos_v7_<año>/<CNN-SNP o CNN-SNP-OHE>
  - imágenes:    DLNN-DM6AUC de v7                      -> Pesos_v7_<año>/DLNN-DM6AUC
Fusión pequeña (fusion_V6: 1-2 capas de 8-32, BatchNorm opcional) con el protocolo v6 (lr 1e-4-5e-3,
paciencia 80, proyección {0, 8, 16}); 100 trials. Trimodal + 3 bimodales.
Usa los pesos 'interno' para la búsqueda y 'completo' para el examen final, igual que v6 y v7.

Uso (en /workspace/trigo_v6, dentro del contenedor), los dos años en paralelo:
    nohup bash lanzar_final.sh Y21 > /dev/null 2>&1 &
    nohup bash lanzar_final.sh Y22 > /dev/null 2>&1 &
Salida: Resultados_final_<año>/hibridos/  (métricas y predicciones), log_final_<año>.txt
"""
import json
import os

import correr_todo_v7 as ct          # datos, utilidades de tareas y reinicio de proceso

AÑO = ct.AÑO
N_TRIALS = 100
FUSION, PROTOCOLO_FUSION = "fusion_V6", "v6"

# nombres propios (no toca v6 ni v7)
ct.SUF, ct.RES = f"_final_{AÑO}", f"Resultados_final_{AÑO}"

with open(f"Resultados_v7_{AÑO}/ramas_elegidas.json") as fh:
    cnn = json.load(fh)["snp"]                                   # CNN-SNP o CNN-SNP-OHE (elegida en v7)
RAMAS = {
    "a":   ("AO",                     f"Pesos_v6_{AÑO}/DLNN-AO"),
    "snp": (ct.CFG[cnn][1],           f"Pesos_v7_{AÑO}/{cnn}"),
    "dm":  ("DM6AUC",                 f"Pesos_v7_{AÑO}/DLNN-DM6AUC"),
}
DEP = ct.CFG[cnn][7]


def hibrido(ramas, y, X, part):
    nombre = "Hibrido_" + "_".join(RAMAS[r][0] for r in ramas)
    for r in ramas:
        for fase in ("interno", "completo"):
            assert os.path.isdir(os.path.join(RAMAS[r][1], fase)), f"No existen los pesos {RAMAS[r][1]}/{fase}"

    def fn():
        ct.ph.hibrido_v2(y_data=y, df_maestro=part, X_ramas={r: X[RAMAS[r][0]] for r in ramas},
                         carpetas_pesos={r: RAMAS[r][1] for r in ramas},
                         nombre_estudio=nombre + ct.SUF, nombre_bd=ct.bd(nombre), funcion_dinamica=FUSION,
                         n_trials=N_TRIALS, cnn_snp=("snp" in ramas), dep=DEP if "snp" in ramas else 1,
                         output_folder=f"{ct.RES}/hibridos", protocolo=PROTOCOLO_FUSION)
        ct.continuar_en_otro_proceso(nombre + ct.SUF, ct.bd(nombre), N_TRIALS)
    ct.tarea(nombre, fn)


def correr():
    y, X, part = ct.cargar()
    print(f"[final {AÑO}] ramas: " + " | ".join(f"{r}: {v[1]}" for r, v in RAMAS.items()))
    os.makedirs(ct.RES, exist_ok=True)
    with open(f"{ct.RES}/ramas.json", "w") as fh:
        json.dump({r: v[1] for r, v in RAMAS.items()} | {"fusion": FUSION, "trials": N_TRIALS}, fh, indent=1)
    for ramas in (("a", "snp", "dm"), ("a", "snp"), ("a", "dm"), ("snp", "dm")):
        hibrido(ramas, y, X, part)
    ct.avisar(f"[final {AÑO}] HÍBRIDOS COMPLETOS")


if __name__ == "__main__":
    correr()
