"""Preparación de datos del artículo (v6) para los dos años, con el mismo proceso.

Año 2020-21 (Y21): 20 ensayos, ambiente B5IR.   Año 2021-22 (Y22): solo B5IR (único ambiente con imágenes).
- Marcadores: archivos imputados del CIMMYT con el MISMO filtro en ambos años (30 % faltantes, MAF 5 %, 90 % genotipado,
  heterocigosidad ≤ 10 %); después de quedarse con las líneas en común se vuelve a aplicar MAF ≥ 5 %.
- Líneas: marcadores ∩ pedigrí ∩ rendimiento (∩ imágenes en Y22).
- Fenómicos por línea (promedio de sus parcelas, sin BLUE):
    DM      = 6 bandas de la última fecha
    DM6     = 6 bandas × todas las fechas, sin procesar
    DM6-AUC = 32 métricas de todas las fechas (AUC, tasa de crecimiento, tasa de senescencia y valor pico de 8 variables:
              ndvi, ndre, gndvi, rededge, nir, red, green, blue), con la misma función de A3-filtro_IMG_kfold.
- Pliegues: 5, estratificados por ensayo, semilla 24591, rotando el pliegue inicial en cada ensayo.
Uso:  python preparar_datos.py <carpeta_del_proyecto>
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

BANDAS = ["blue", "green", "ndvi", "nir", "red", "rededge"]
SEMILLA_FOLDS = 24591
ARCH = {
    "Y21": dict(snp="Y20-21/Numerical_markers_30MISS_5MAF_90GENO_Y20-21.txt", A="Y20-21/A_matrix_Y20-21.txt",
                yld="Y20-21/Yield_Y20-21.xlsx", img="Y20-21/Images_Y20-21.xlsx"),
    "Y22": dict(snp="Y21-22/Numerical_markers_Y21-22_30MISS_5MAF_90GENO_10HET_imputed.txt", A="Y21-22/A_matrix_EYT_21-22.txt",
                yld="Y21-22/Yield_Y21-22.xlsx", img="Y21-22/Images_Y21-22.xlsx"),
}


# ------------------------------------------------------------------ lectura y estandarización de nombres
def leer_rendimiento(año, ruta):
    y = pd.read_excel(ruta)
    if año == "Y21":
        y = y.rename(columns={"trial.no": "ensayo", "rep": "rep", "subblock": "bloque", "row": "fila", "column": "columna"})
    else:
        y = y[y["env"] == "B5IR"].copy()
        y["ensayo"] = y["trial.id"].str[-2:].astype(int)
        y = y.rename(columns={"Rep": "rep", "Sub_block": "bloque", "Row": "fila", "Column": "columna"})
    y["ton.ha"] = pd.to_numeric(y["ton.ha"], errors="coerce")
    return y[["plot", "GID", "ensayo", "rep", "bloque", "fila", "columna", "ton.ha"]].reset_index(drop=True)


def leer_imagenes(año, ruta):
    """Devuelve una tabla larga: plot, GID_img, fecha (datetime), banda, valor (promedio de las lecturas de la parcela)."""
    if año == "Y21":
        im = pd.read_excel(ruta)
        im = im.rename(columns={"Band": "banda", "Mean": "valor", "GID": "GID_img"})
        if "GID_img" not in im.columns:                 # el archivo de 2020-21 no trae GID: se toma del rendimiento por parcela
            im["GID_img"] = -1
        im["fecha"] = pd.to_datetime(im["Date"].astype(str), format="%Y%m%d")
    else:
        raw = pd.read_excel(ruta, header=None)
        fechas, hdr = raw.iloc[0].tolist(), raw.iloc[1].tolist()
        datos = raw.iloc[2:].reset_index(drop=True)
        base = pd.DataFrame({"plot": pd.to_numeric(datos[1]), "GID_img": pd.to_numeric(datos[2])})
        partes, actual = [], None
        for j in range(5, raw.shape[1]):
            if isinstance(fechas[j], (pd.Timestamp, datetime.datetime)):
                actual = pd.Timestamp(fechas[j])
            b = str(hdr[j]).lower().replace("_", "")
            partes.append(base.assign(fecha=actual, banda=b, valor=pd.to_numeric(datos[j], errors="coerce")))
        im = pd.concat(partes, ignore_index=True)
    im["banda"] = im["banda"].str.lower().str.replace("_", "", regex=False)
    return im.groupby(["plot", "GID_img", "fecha", "banda"], as_index=False)["valor"].mean()


def metricas_fenologicas(ancho):
    """Misma función que A3-filtro_IMG_kfold: AUC (trapecios sobre el día del año), valor pico,
    tasa de crecimiento (inicio → pico) y de senescencia (pico → final), por parcela."""
    w = ancho.copy()
    w["ndre"] = (w["nir"] - w["rededge"]) / (w["nir"] + w["rededge"] + 1e-6)
    w["gndvi"] = (w["nir"] - w["green"]) / (w["nir"] + w["green"] + 1e-6)
    w["doy"] = w["fecha"].dt.dayofyear
    filas = []
    for plot, g in w.sort_values(["plot", "doy"]).groupby("plot"):
        x = g["doy"].to_numpy(float); fila = {"plot": plot}
        for col in ["ndvi", "ndre", "gndvi", "rededge", "nir", "red", "green", "blue"]:
            yv = g[col].to_numpy(float); k = int(np.argmax(yv))
            fila[f"{col}_auc"] = float(np.sum((yv[:-1] + yv[1:]) * np.diff(x) / 2))
            fila[f"{col}_tasa_crecimiento"] = (yv[k] - yv[0]) / (x[k] - x[0]) if k > 0 else 0.0
            fila[f"{col}_tasa_senescencia"] = (yv[-1] - yv[k]) / (x[-1] - x[k]) if k < len(yv) - 1 else 0.0
            fila[f"{col}_valor_pico"] = yv[k]
        filas.append(fila)
    return pd.DataFrame(filas)


def pliegues(gids, ensayo):
    rng = np.random.default_rng(SEMILLA_FOLDS)
    f = pd.Series(0, index=gids)
    grupos = pd.Series(ensayo, index=gids)
    for j, (_, idx) in enumerate(grupos.groupby(grupos).groups.items()):
        idx = list(idx)
        f.loc[idx] = rng.permutation(np.resize(np.roll(np.arange(1, 6), -j), len(idx)))
    return f.values


# ------------------------------------------------------------------ un año completo
def preparar(año, carpeta, salida="Datos_entrenamiento"):
    a = {k: os.path.join(carpeta, v) for k, v in ARCH[año].items()}
    R = {"año": año}
    snp = pd.read_csv(a["snp"], sep="\t", index_col=0); snp.index = snp.index.astype(int)
    A = pd.read_csv(a["A"], sep="\t", index_col=0); A.index = A.index.astype(int); A.columns = A.columns.astype(int)
    yld = leer_rendimiento(año, a["yld"])
    im = leer_imagenes(año, a["img"])
    R.update(snp_crudo=snp.shape, lineas_A=A.shape[0], parcelas=len(yld), parcelas_sin_rend=int(yld["ton.ha"].isna().sum()),
             lineas_rend=yld.GID.nunique(), ensayos=yld.ensayo.nunique(), fechas=[d.strftime("%Y-%m-%d") for d in sorted(pd.to_datetime(im["fecha"].unique()))])

    # imágenes: empatar por parcela Y por GID
    gid_img = im.drop_duplicates("plot").set_index("plot")["GID_img"]
    gid_rend = yld.set_index("plot")["GID"]
    comun = gid_rend.index.intersection(gid_img.index)
    sin_gid = gid_img.loc[comun].values < 0
    ok = comun[(gid_rend.loc[comun].values == gid_img.loc[comun].values) | sin_gid]
    R.update(parcelas_con_imagen=len(comun), parcelas_GID_coincide=len(ok), parcelas_GID_distinto=len(comun) - len(ok))
    im = im[im["plot"].isin(ok)]
    ancho = im.pivot_table(index=["plot", "fecha"], columns="banda", values="valor").reset_index()
    ultima = ancho.fecha.max()
    por_parcela = {
        "DM": ancho[ancho.fecha == ultima].set_index("plot")[BANDAS],
        "DM6": ancho.assign(f=ancho.fecha.dt.strftime("%Y%m%d")).pivot(index="plot", columns="f", values=BANDAS)
                    .pipe(lambda d: d.set_axis([f"{b}_{f}" for b, f in d.columns], axis=1)),
        "DM6AUC": metricas_fenologicas(ancho).set_index("plot"),
    }
    plot_gid = yld.set_index("plot")["GID"]

    # líneas en común
    lineas_img = set(plot_gid.loc[ok])
    gids = sorted(set(snp.index) & set(A.index) & set(yld.dropna(subset=["ton.ha"]).GID) & lineas_img)
    snp_f = snp.loc[gids]
    p = snp_f.values.mean(0) / 2; maf = np.minimum(p, 1 - p); het = (snp_f.values == 1).mean(0)
    keep = (maf >= 0.05) & (het <= 0.10)
    snp_f = snp_f.loc[:, keep]
    R.update(lineas=len(gids), snp_final=int(keep.sum()), snp_quitados_MAF=int((maf < 0.05).sum()), het_max=float(het.max()))

    ens = yld.groupby("GID")["ensayo"].agg(lambda t: t.iloc[0] if t.nunique() == 1 else -1).reindex(gids).astype(int)
    folds = pliegues(np.array(gids), ens.values)
    tab = pd.DataFrame({"GID": gids, "trial.no": ens.values, "Fold_Prueba": folds})
    R.update(pliegues=tab.Fold_Prueba.value_counts().sort_index().tolist(),
             testigos=list(tab.GID[tab["trial.no"] < 0]),
             lineas_por_ensayo_pliegue=(lambda x: (int(x.values.min()), int(x.values.max())))(
                 tab[tab["trial.no"] > 0].groupby(["trial.no", "Fold_Prueba"]).size().unstack(fill_value=0)))

    os.makedirs(os.path.join(carpeta, salida), exist_ok=True)
    s = lambda n: os.path.join(carpeta, salida, f"{n}_{año}.csv")
    snp_f.rename_axis("GID").reset_index().to_csv(s("SNP"), index=False)
    A.loc[gids, gids].rename_axis("GID").reset_index().to_csv(s("X_A"), index=False)
    tab.to_csv(s("folds"), index=False)
    for nombre, d in por_parcela.items():
        prom = d.join(plot_gid).groupby("GID").mean().loc[gids]
        assert prom.notna().all().all(), f"{nombre}: hay faltantes"
        prom.rename_axis("GID").reset_index().to_csv(s(f"IMG_{nombre}"), index=False)
        R[f"{nombre}_variables"] = prom.shape[1]
    # rendimiento por parcela para los BLUEs (todas las parcelas del diseño)
    yld.to_csv(s("rendimiento_parcelas"), index=False)

    # resumen del rendimiento
    yl = yld.dropna(subset=["ton.ha"])
    r1 = yl[yl.rep == 1].drop_duplicates(["ensayo", "GID"]).set_index(["ensayo", "GID"])["ton.ha"]
    r2 = yl[yl.rep == 2].drop_duplicates(["ensayo", "GID"]).set_index(["ensayo", "GID"])["ton.ha"]
    i = r1.index.intersection(r2.index)
    R.update(rend_media=round(yl["ton.ha"].mean(), 2), rend_de=round(yl["ton.ha"].std(), 2),
             rend_min=round(yl["ton.ha"].min(), 2), rend_max=round(yl["ton.ha"].max(), 2),
             repetibilidad_rep1_rep2=round(float(np.corrcoef(r1[i], r2[i])[0, 1]), 3))
    return R


if __name__ == "__main__":
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "."
    for año in ["Y21", "Y22"]:
        print(f"\n===== {año} =====")
        for k, v in preparar(año, carpeta).items():
            print(f"  {k}: {v}")
