"""Desglose de líneas por fuente de datos (revisor 1, comentario menor 4). SOLO LEE los datos crudos; no escribe nada.
Uso (en /workspace/trigo_v6):   python desglose_lineas.py . | tee logs/desglose_lineas.txt"""
import sys

import pandas as pd

import preparar_datos as pd_

carpeta = sys.argv[1] if len(sys.argv) > 1 else "."
for año in ("Y21", "Y22"):
    a = {k: f"{carpeta}/{v}" for k, v in pd_.ARCH[año].items()}
    snp = pd.read_csv(a["snp"], sep="\t", index_col=0); snp.index = snp.index.astype(int)
    A = pd.read_csv(a["A"], sep="\t", index_col=0); A.index = A.index.astype(int)
    yld = pd_.leer_rendimiento(año, a["yld"])
    im = pd_.leer_imagenes(año, a["img"])
    gid_img = im.drop_duplicates("plot").set_index("plot")["GID_img"]
    gid_rend = yld.set_index("plot")["GID"]
    comun = gid_rend.index.intersection(gid_img.index)
    ok = comun[(gid_rend.loc[comun].values == gid_img.loc[comun].values) | (gid_img.loc[comun].values < 0)]
    S = {"evaluadas": set(yld.GID), "rendimiento": set(yld.dropna(subset=["ton.ha"]).GID), "SNP": set(snp.index),
         "pedigri": set(A.index), "imagenes": set(gid_rend.loc[ok])}
    ev = S["evaluadas"]
    final = S["rendimiento"] & S["SNP"] & S["pedigri"] & S["imagenes"]
    print(f"\n===== {año} =====")
    print(f"  líneas evaluadas (con parcelas): {len(ev)}   | parcelas: {len(yld)}   | líneas en archivo SNP: {len(S['SNP'])}"
          f"   | líneas en matriz A: {len(S['pedigri'])}")
    for k in ("rendimiento", "SNP", "pedigri", "imagenes"):
        print(f"  evaluadas SIN {k:12s}: {len(ev - S[k])}")
    solo = {k: len([g for g in ev - final if [j for j in ('rendimiento', 'SNP', 'pedigri', 'imagenes') if g not in S[j]] == [k]])
            for k in ("rendimiento", "SNP", "pedigri", "imagenes")}
    print(f"  excluidas en total: {len(ev - final)}  | por una sola causa: {solo}  | por varias causas: {len(ev - final) - sum(solo.values())}")
    print(f"  líneas finales con las cuatro fuentes: {len(final)}")
    print(f"  SNP en el archivo del CIMMYT: {snp.shape[1]}  | % de genotipos faltantes en ese archivo: {100 * snp.isna().values.mean():.3f}")
