"""
Tablas y figuras del artículo a partir de las predicciones guardadas (no reentrena nada).

Lee (en esta carpeta):
  Datos_entrenamiento/y_folds_<año>.csv                          pliegues y BLUEs (R1)
  Resultados_R_v6/pred_<año>_<modelo>.csv                         BLUP y RKHS (R2 y R3)
  Resultados_v6_<año>/unimodales/*_predicciones.csv               redes, espacio compacto (predicciones_unimodales_v6.py)
  Resultados_v7_<año>/unimodales/*_predicciones.csv               redes, espacio amplio  (predicciones_unimodales_v7.py)
  Resultados_{v6,v7,final}_<año>/hibridos/*_predicciones.csv      híbridos
  Resultados_v7_<año>/montesinos/*_predicciones.csv               MMDL
  resumen_optuna.csv                                              (exportar_optuna.py)
  parametros_unimodales.csv                                       (contar_parametros.py)
Escribe:
  Tablas_articulo.csv                 r, RMSE, MAE (media y DE de 5 pliegues) y t pareada contra la referencia
  Supplementary_Tables_1-3.xlsx       1 todos los modelos, 2 métricas por pliegue, 3 Optuna (cuatro paneles)
  Figura3_observado_predicho.png      RKHS trimodal e híbrido final, dos ciclos
En el artículo, "DMT" son las 32 métricas temporales (DM6AUC en los archivos) y "DM" las bandas de la última fecha.
Uso:  python tablas_figuras_articulo.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel

AÑOS = ("Y21", "Y22")
CICLO = {"Y21": "2020–2021", "Y22": "2021–2022"}
TRI = "RKHS_tri_A+SNP+DM6AUC"


# ------------------------------------------------------------------ lectura
def leer(f):
    d = pd.read_csv(f)
    d.columns = [c.lower() for c in d.columns]
    d = d.rename(columns={"y_real": "y", "y_pred": "p"})
    d["gid"] = d["gid"].astype(str)
    return d[["gid", "y", "p"]]


def fuentes(a):
    F = {}
    for f in glob.glob(f"Resultados_R_v6/pred_{a}_*.csv"):
        F[("R", os.path.basename(f)[len(f"pred_{a}_"):-4])] = f
    for v in ("v6", "v7"):
        for f in glob.glob(f"Resultados_{v}_{a}/unimodales/*_predicciones.csv"):
            F[(v, os.path.basename(f)[:-len("_predicciones.csv")])] = f
    for v in ("v6", "v7", "final"):
        for f in glob.glob(f"Resultados_{v}_{a}/hibridos/*_predicciones.csv") + \
                 glob.glob(f"Resultados_{v}_{a}/montesinos/*_predicciones.csv"):
            F[(v, os.path.basename(f).split(f"_{v}_")[0])] = f
    return F


FOLD, filas = {}, []
for a in AÑOS:
    yf = pd.read_csv(f"Datos_entrenamiento/y_folds_{a}.csv"); yf["GID"] = yf.GID.astype(str)
    for (fuente, m), f in sorted(fuentes(a).items()):
        d = leer(f).merge(yf[["GID", "Fold_Prueba", "y"]].rename(columns={"y": "yref"}), left_on="gid", right_on="GID")
        assert len(d) == len(yf) and np.allclose(d.y, d.yref, atol=1e-3), (a, fuente, m)
        g = d.groupby("Fold_Prueba")
        met = dict(r=g.apply(lambda x: pearsonr(x.y, x.p)[0]).values,
                   rmse=g.apply(lambda x: np.sqrt(np.mean((x.y - x.p) ** 2))).values,
                   mae=g.apply(lambda x: np.mean(np.abs(x.y - x.p))).values)
        FOLD[(a, fuente, m)] = met
        filas.append(dict(año=a, fuente=fuente, modelo=m, r=met["r"].mean(), sd_r=met["r"].std(ddof=1),
                          RMSE=met["rmse"].mean(), sd_RMSE=met["rmse"].std(ddof=1),
                          MAE=met["mae"].mean(), sd_MAE=met["mae"].std(ddof=1)))
T = pd.DataFrame(filas)

# ------------------------------------------------------------------ referencia de cada modelo y t pareada
REF_UNI = {"DLNN-AO": "BLUP_A", "DLNN-GO": "GBLUP_SNP", "DLNN-SNP": "RKHS_SNP", "CNN-SNP": "RKHS_SNP", "CNN-SNP-OHE": "RKHS_SNP",
           "DLNN-DM6AUC": "RKHS_DM6AUC", "DLNN-DM": "RKHS_DM", "DLNN-DM6": "RKHS_DM6"}
for i, f in T.iterrows():
    ref = None
    if f.fuente in ("v6", "v7") and f.modelo in REF_UNI:
        ref = REF_UNI[f.modelo]
    elif f.modelo.startswith(("Hibrido", "MMDL")):
        ref = "RKHS_A+SNP" if f.modelo.endswith(("_AO_SNP", "_AO_SNP-OHE")) else TRI
    if ref and (f.año, "R", ref) in FOLD:
        x, z = FOLD[(f.año, f.fuente, f.modelo)]["r"], FOLD[(f.año, "R", ref)]["r"]
        T.loc[i, ["ref", "dif_r", "p_r"]] = [ref, (x - z).mean(), ttest_rel(x, z).pvalue]
T.round(4).to_csv("Tablas_articulo.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
print(T.round(3).to_string(index=False))

# ------------------------------------------------------------------ material suplementario (Excel)
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

NOM = {"BLUP_A": ("BLUP", "A"), "GBLUP_SNP": ("GBLUP", "SNP"), "RKHS_SNP": ("RKHS", "SNP"), "BLUP_DM6AUC": ("BLUP", "DMT"),
       "RKHS_DM6AUC": ("RKHS", "DMT"), "RKHS_DM6": ("RKHS", "Raw bands, all dates"), "RKHS_DM": ("RKHS", "DM"),
       TRI: ("RKHS", "A + SNP + DMT"), "BLUP_tri_A+SNP+DM6AUC": ("BLUP", "A + SNP + DMT"),
       "RKHS_A+SNP": ("RKHS", "A + SNP"), "BLUP_A+SNP": ("BLUP", "A + SNP"),
       "DLNN-AO": ("DLNN", "AO"), "DLNN-GO": ("DLNN", "GO"), "DLNN-SNP": ("DLNN", "SNP"),
       "CNN-SNP": ("1D-CNN", "SNP"), "CNN-SNP-OHE": ("1D-CNN", "SNP-OHE"),
       "DLNN-DM6AUC": ("DLNN", "DMT"), "DLNN-DM6": ("DLNN", "Raw bands, all dates"), "DLNN-DM": ("DLNN", "DM"),
       "MMDL_Montesinos": ("MMDL (Montesinos-López et al., 2023)", "AO + GO + DMT")}
ESP = {"R": "–", "v6": "Compact", "v7": "Wide", "final": "Hybrid (compact AO + wide SNP and DMT)"}


def nombre(m):
    if m.startswith("Hibrido"):
        ent = m.replace("Hibrido_", "").replace("DM6AUC", "DMT").replace("_", " + ")
        if m.endswith("_DM6"):
            ent = ent[:-3] + "raw bands (all dates)"
        return "Hybrid", ent
    return NOM.get(m, (m, ""))


# --- Tabla 1: todos los modelos
s1 = T.copy()
s1["Cycle"] = s1.año.map(CICLO)
s1["Model"], s1["Input"] = zip(*[nombre(m) for m in s1.modelo])
s1["Search space"] = s1.fuente.map(ESP)
s1.loc[s1.modelo == "MMDL_Montesinos", "Search space"] = "Original ranges (150 trials)"
s1["Paired reference"] = s1.get("ref", pd.Series(dtype=str)).map(lambda r: "" if pd.isna(r) else " ".join(nombre(r)))
s1 = s1[["Cycle", "Model", "Input", "Search space", "r", "sd_r", "RMSE", "sd_RMSE", "MAE", "sd_MAE",
         "Paired reference", "dif_r", "p_r"]]
s1.columns = ["Cycle", "Model", "Input", "Search space", "r", "SD r", "RMSE", "SD RMSE", "MAE", "SD MAE",
              "Paired reference", "Δr vs reference", "P (paired t)"]
_orden = {"–": 0, "Compact": 1, "Wide": 2}
s1 = s1.sort_values(["Cycle", "Search space", "Model", "Input"],
                    key=lambda c: c.map(_orden).fillna(3) if c.name == "Search space" else c)

# --- Tabla 2: métricas por pliegue
s2 = pd.DataFrame([dict(Cycle=CICLO[a], Model=" ".join(nombre(m)), **{"Search space": ESP[f]}, Fold=k + 1,
                        r=v["r"][k], RMSE=v["rmse"][k], MAE=v["mae"][k])
                   for (a, f, m), v in FOLD.items() for k in range(5)]).sort_values(["Cycle", "Search space", "Model", "Fold"])


# --- Tabla 3: estudios de Optuna en cuatro paneles (un panel por tipo de red, una columna por hiperparámetro)
def _g(v):
    return f"{float(v):.2g}"


def _lista(d, patron, n, penal=False):
    return ", ".join((_g(d[patron.format(i)]) if penal else f"{d[patron.format(i)]:.2f}" if isinstance(d[patron.format(i)], float)
                      else str(d[patron.format(i)])) for i in range(n) if patron.format(i) in d)


def _sn(x):
    return "Yes" if x else "No"


def h_dlnn(d, ver):
    n = d["n_layers"]
    act = "ReLU (fixed)" if ver == "v6" else ("ReLU" if d.get("activacion") == "relu" else f"LeakyReLU (slope {d['leaky_alpha']:.2f})")
    return {"Learning rate": _g(d["learning_rate"]), "Batch size": d["batch_size"], "Hidden layers": n,
            "Units per layer": _lista(d, "units_l{}", n), "Dropout per layer": _lista(d, "drop_l{}", n),
            "L2 penalty per layer": _lista(d, "l2_l{}", n, True),
            "Batch normalization": _sn(d["batchnorm"]) if ver == "v6" else "Yes (fixed)", "Activation": act}


def h_cnn(d, ver):
    nc, nd = d["n_conv_layers"], d["n_dense_layers"]
    if ver == "v6":
        return {"Learning rate": _g(d["learning_rate"]), "Batch size": d["batch_size"], "Convolutional layers": nc,
                "Filters per conv. layer": _lista(d, "conv{}_filtros", nc),
                "Kernel size per conv. layer": ", ".join([str(d["conv_kernel"])] * nc), "Dropout per conv. layer": "–",
                "Pool size": d["pool_size"], "Dense layers": nd, "Units per dense layer": _lista(d, "dense{}_units", nd),
                "Dropout per dense layer": ", ".join([f"{d['drop']:.2f}"] * nd), "L1 penalty per dense layer": "–",
                "L2 penalty per dense layer": ", ".join([_g(d["l2"])] * nd), "Batch normalization": _sn(d["batchnorm"]),
                "Activation": "ReLU (fixed)"}
    return {"Learning rate": _g(d["learning_rate"]), "Batch size": d["batch_size"], "Convolutional layers": nc,
            "Filters per conv. layer": _lista(d, "conv{}_filtros", nc), "Kernel size per conv. layer": _lista(d, "conv{}_kernel", nc),
            "Dropout per conv. layer": _lista(d, "conv{}_drop", nc), "Pool size": d["pool_size"], "Dense layers": nd,
            "Units per dense layer": _lista(d, "dense{}_units", nd), "Dropout per dense layer": _lista(d, "dense{}_drop", nd),
            "L1 penalty per dense layer": _lista(d, "dense{}_l1", nd, True),
            "L2 penalty per dense layer": _lista(d, "dense{}_l2", nd, True), "Batch normalization": "Yes (fixed)",
            "Activation": f"LeakyReLU (slope {d['leaky_alpha']:.2f})"}


def h_fus(d, ver):
    n = d["fusion_layers"]
    return {"Learning rate": _g(d["learning_rate"]), "Batch size": d["batch_size"],
            "Projection of each branch (units)": "None" if d["proj_dim"] == 0 else d["proj_dim"], "Fusion layers": n,
            "Units per layer": _lista(d, "fusion_units_l{}", n), "Dropout per layer": _lista(d, "fusion_drop_l{}", n),
            "L2 penalty": _g(d["fusion_l2"]),
            "Batch normalization": _sn(d["fusion_batchnorm"]) if "fusion_batchnorm" in d else "Yes (fixed)"}


def h_mmdl(d, ver):
    return {"Learning rate": _g(d["learning_rate"]), "Learning-rate decay": f"{d['decay']:.3f}", "L2 penalty": _g(d["l2"]),
            "Dropout": f"{d['dropout']:.2f}", "Pedigree subnetwork: layers": d["a_capas"],
            "Pedigree subnetwork: units (first layer)": d["a_n1"], "Genomic subnetwork: layers": d["go_capas"],
            "Genomic subnetwork: units (first layer)": d["go_n1"], "Multispectral subnetwork: layers": d["dm_capas"],
            "Multispectral subnetwork: units (first layer)": d["dm_n1"]}


def n_parametros(estudio, uni):
    """(total, entrenados en la última etapa): redes individuales de parametros_unimodales.csv; híbridos y MMDL de sus métricas."""
    if estudio in uni:
        return uni[estudio], uni[estudio]
    for f in glob.glob(f"Resultados_*/*/{estudio}_metricas.csv"):
        m = pd.read_csv(f)
        if "n_param_total" in m:
            return int(m.n_param_total.iloc[0]), int(m.n_param_entrenables.iloc[0])
        if "n_parametros" in m:
            return int(m.n_parametros.iloc[0]), int(m.n_parametros.iloc[0])
    return "", ""


TIT = {1: "Supplementary Table 1. Predictive performance of all models in both breeding cycles (mean and SD across five outer test "
          "folds; paired t-test of fold-level r against the statistical reference with the same data sources).",
       2: "Supplementary Table 2. Fold-level metrics of every model (outer test folds; identical fold assignments for all models).",
       3: "Supplementary Table 3. Optuna studies: trials, internal-validation MSE of the selected configuration (standardized scale), "
          "search time, number of parameters and selected hyperparameters of every network and fusion block, in four panels by type "
          "of network. Per-layer values are listed from the first to the last layer. For the hybrids, the total number of "
          "parameters includes the frozen branches and only the fusion block is trained in the last stage."}
paneles = []
if os.path.exists("resumen_optuna.csv"):
    o = pd.read_csv("resumen_optuna.csv")
    o["Cycle"] = o.estudio.str[-3:].map(CICLO)
    o["ver"] = o.estudio.str.extract(r"_(v6|v7|final)_")[0]
    o["mod"] = o.estudio.str.replace(r"_(v6|v7|final)_Y2\d", "", regex=True)
    o["Model"] = [" ".join(nombre(m)) for m in o["mod"]]
    o["Search space"] = o.ver.map(ESP)
    o.loc[o["mod"] == "MMDL_Montesinos", "Search space"] = "Original ranges"
    uni = pd.read_csv("parametros_unimodales.csv").set_index("estudio")["n_param"].to_dict() if os.path.exists("parametros_unimodales.csv") else {}
    o["npar"], o["nent"] = zip(*[n_parametros(e, uni) for e in o.estudio])
    o["tipo"] = ["fus" if e.startswith("Hibrido") else "mmdl" if e.startswith("MMDL") else "cnn" if e.startswith("CNN") else "dlnn"
                 for e in o.estudio]
    COMUN = {"Cycle": "Cycle", "Model": "Model", "Search space": "Search space", "trials": "Trials",
             "completos": "Completed (not pruned)", "mejor_trial": "Best trial", "val_mse": "Internal-validation MSE (standardized)",
             "horas_suma_trials": "Search time (h, sum of trials)", "npar": "Parameters (total)",
             "nent": "Parameters trained in the last stage"}
    for tipo, titulo, fn in (("dlnn", "A. Fully connected networks (DLNN), unimodal", h_dlnn),
                             ("cnn", "B. One-dimensional convolutional networks (1D-CNN), unimodal", h_cnn),
                             ("fus", "C. Fusion block of the hybrids (branches frozen)", h_fus),
                             ("mmdl", "D. Jointly trained multimodal model (MMDL)", h_mmdl)):
        sub = o[o.tipo == tipo].sort_values(["Cycle", "Search space", "Model"])
        hp = pd.DataFrame([fn(json.loads(pp), v) for pp, v in zip(sub.params, sub.ver)], index=sub.index)
        paneles.append((titulo, pd.concat([sub[list(COMUN)].rename(columns=COMUN), hp], axis=1)))


def _formato(ws, fila_enc, df):
    for c in ws[fila_enc]:
        c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="top")
    for jj, col in enumerate(df.columns, 1):
        ancho = max(len(str(col)) * 0.55, *(len(str(x)) for x in df[col].head(200))) if len(df) else len(col)
        act = ws.column_dimensions[get_column_letter(jj)].width or 0
        ws.column_dimensions[get_column_letter(jj)].width = max(act, min(max(9, ancho * 1.05), 34))


with pd.ExcelWriter("Supplementary_Tables_1-3.xlsx", engine="openpyxl") as w:
    for k, df in ((1, s1), (2, s2)):
        hoja = f"Supplementary Table {k}"
        df.round(4).to_excel(w, sheet_name=hoja, index=False, startrow=2)
        ws = w.sheets[hoja]
        ws["A1"] = TIT[k]; ws["A1"].font = Font(bold=True)
        _formato(ws, 3, df); ws.freeze_panes = "A4"
    if paneles:
        hoja, fila = "Supplementary Table 3", 2
        for titulo, df in paneles:
            df.round(4).to_excel(w, sheet_name=hoja, index=False, startrow=fila + 1)
            ws = w.sheets[hoja]
            ws.cell(row=fila + 1, column=1, value=titulo).font = Font(bold=True, size=12)
            _formato(ws, fila + 2, df)
            for r_ in ws.iter_rows(min_row=fila + 3, max_row=fila + 2 + len(df)):
                for c in r_:
                    c.alignment = Alignment(horizontal="left", vertical="top")
            fila += len(df) + 4
        ws["A1"] = TIT[3]; ws["A1"].font = Font(bold=True)

# ------------------------------------------------------------------ Figura 3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ["#2B5D8A", "#C26A1B", "#2E7D4F", "#A23B3B", "#6A4C93"]
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.6), dpi=300)
for i, a in enumerate(AÑOS):
    yf = pd.read_csv(f"Datos_entrenamiento/y_folds_{a}.csv"); yf["GID"] = yf.GID.astype(str)
    for j, (tit, f) in enumerate((("Multimodal RKHS (A + SNP + DMT)", f"Resultados_R_v6/pred_{a}_{TRI}.csv"),
                                  ("Trimodal hybrid (AO + SNP + DMT)",
                                   f"Resultados_final_{a}/hibridos/Hibrido_AO_SNP_DM6AUC_final_{a}_predicciones.csv"))):
        d = leer(f).merge(yf[["GID", "Fold_Prueba"]], left_on="gid", right_on="GID")
        ax = axs[i, j]
        for k in sorted(d.Fold_Prueba.unique()):
            s = d[d.Fold_Prueba == k]
            ax.scatter(s.p, s.y, s=7, alpha=0.6, color=COL[k - 1], lw=0, label=f"Fold {k}")
        rf = [pearsonr(g.y, g.p)[0] for _, g in d.groupby("Fold_Prueba")]
        lo, hi = min(d.y.min(), d.p.min()) - 0.1, max(d.y.max(), d.p.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="#555555")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.text(0.03, 0.97, f"r (fold mean ± SD) = {np.mean(rf):.3f} ± {np.std(rf, ddof=1):.3f}\n"
                            f"r (pooled) = {pearsonr(d.y, d.p)[0]:.3f}", transform=ax.transAxes, va="top",
                fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
        ax.set_title(f"{tit}\n{CICLO[a]}", fontsize=8.5)
        ax.set_xlabel("Predicted BLUE (Mg ha$^{-1}$)"); ax.set_ylabel("Observed BLUE (Mg ha$^{-1}$)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
axs[0, 1].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc="lower right", markerscale=1.8)
fig.tight_layout(); fig.savefig("Figura3_observado_predicho.png", facecolor="white")
print("\nListo: Tablas_articulo.csv, Supplementary_Tables_1-3.xlsx y Figura3_observado_predicho.png")
