import pickle, re
import numpy as np
import pandas as pd

# --- 1. Embeddings TITAN ---
with open("data/TCGA_TITAN_features.pkl", "rb") as f:
    d = pickle.load(f)
emb = d["embeddings"]
slides = pd.Series([str(x).split("/")[-1] for x in d["filenames"]])
print("Ejemplos filenames:", slides.head(3).tolist())
print("NaN en embeddings:", np.isnan(emb).sum())

meta = pd.DataFrame({
    "slide": slides,
    "patient": slides.str[:12],
    "sample_type": slides.str[13:15],
    "slide_type": slides.str.extract(r"-(DX|TS|BS|MS)\d*", expand=False),
})

# --- 2. GISTIC HNSC ---
g = pd.read_csv("data/HNSC_gistic2_thresholded.tsv.gz", sep="\t", index_col=0)
cnv = g.loc[["CCND1", "CTTN", "FGF19", "ANO1"]].T
cnv["patient"] = cnv.index.str[:12]

# --- 3. Clínica ---
c = pd.read_csv("data/HNSC_clinicalMatrix.tsv", sep="\t", index_col=0)
c = c[c.index.str[13:15] == "01"].copy()
c["patient"] = c.index.str[:12]
oral = ["Oral Tongue", "Oral Cavity", "Floor of mouth", "Buccal Mucosa", "Alveolar Ridge", "Hard Palate", "Lip"]
orofar = ["Tonsil", "Base of tongue", "Oropharynx"]
sub = c["anatomic_neoplasm_subdivision"]
c["subsitio"] = np.select([sub.isin(oral), sub.isin(orofar), sub == "Larynx", sub == "Hypopharynx"],
                          ["Cavidad oral", "Orofaringe", "Laringe", "Hipofaringe"], "Otro/NA")

# --- 4. Láminas HNSC ---
hnsc = meta[meta["patient"].isin(cnv["patient"])]
print("\nLáminas HNSC en pkl:", len(hnsc))
print("Tipo de lámina:\n", hnsc["slide_type"].value_counts(dropna=False))
print("Tipo de muestra:\n", hnsc["sample_type"].value_counts())
hnsc = hnsc[(hnsc["sample_type"] == "01") & (hnsc["slide_type"] == "DX")]
print("Láminas por paciente (DX, 01):\n", hnsc.groupby("patient").size().value_counts().sort_index())

# --- 5. Cohorte final ---
coh = (cnv[cnv["patient"].isin(hnsc["patient"])]
       .merge(c[["patient", "subsitio"]].drop_duplicates("patient"), on="patient", how="left"))
coh["CCND1_amp"] = (coh["CCND1"] == 2).astype(int)
print("\nPacientes en cohorte final:", len(coh))
print("Positivos / negativos:\n", coh["CCND1_amp"].value_counts())
print("\nPositivos por subsitio:\n", pd.crosstab(coh["subsitio"], coh["CCND1_amp"], margins=True))

# --- 6. Sanity: co-amplificación 11q13 ---
pos = coh[coh["CCND1_amp"] == 1]
for gen in ["CTTN", "FGF19", "ANO1"]:
    print(f"% de CCND1+ con {gen} +2: {(pos[gen] == 2).mean():.0%}")

coh.to_csv("metadata/cohorte_preliminar.tsv", sep="\t", index=False)
