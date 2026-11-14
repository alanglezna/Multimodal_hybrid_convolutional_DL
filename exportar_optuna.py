"""
Exporta de todos los estudios de Optuna (v6, v7 y final) lo que necesita el artículo:
mejor trial, val MSE, hiperparámetros elegidos, número de trials y tiempo total de búsqueda.
Uso (en /workspace/trigo_v6):  python exportar_optuna.py   ->  resumen_optuna.csv
"""
import glob
import json
import os

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
filas = []
for f in sorted(glob.glob("Opt_*_v6_Y2*.db") + glob.glob("Opt_*_v7_Y2*.db") + glob.glob("Opt_*_final_Y2*.db")):
    if "prueba" in f:
        continue
    url = f"sqlite:///{f}"
    for s in optuna.study.get_all_study_summaries(storage=url):
        st = optuna.load_study(study_name=s.study_name, storage=url)
        ok = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not ok:
            continue
        ini = min(t.datetime_start for t in st.trials if t.datetime_start)
        fin = max(t.datetime_complete for t in st.trials if t.datetime_complete)
        horas_suma = sum((t.datetime_complete - t.datetime_start).total_seconds()
                         for t in st.trials if t.datetime_complete and t.datetime_start) / 3600
        b = st.best_trial
        filas.append(dict(archivo=os.path.basename(f), estudio=s.study_name, trials=len(st.trials),
                          completos=len(ok), mejor_trial=b.number, val_mse=b.value,
                          horas_suma_trials=round(horas_suma, 2),
                          horas_reloj=round((fin - ini).total_seconds() / 3600, 2),
                          params=json.dumps(b.params), attrs=json.dumps(b.user_attrs, default=str)))
pd.DataFrame(filas).to_csv("resumen_optuna.csv", index=False)
print(f"{len(filas)} estudios -> resumen_optuna.csv")
