"""
Pipeline A: Residual-PAR con pseudo-pairing (KNN) senza variabili di contesto.

Flusso:
1) Estrazione "baseline" per ogni paziente (prima visita) su un sottoinsieme di feature scelte
   per il matching (baseline_match_feats).
2) KNN tra baseline dei pazienti Leaspy e baseline dei pazienti reali → per ogni paziente
   Leaspy troviamo k reali più simili.
3) Riallineiamo le sequenze reali dei vicini ai timepoint di ciascun paziente Leaspy per
   ottenere una media "target" realistica.
4) Calcoliamo i RESIDUI: resid = mean(real_neighbors_aligned) - leaspy.
5) Alleniamo PARSynthesizer sui residui (senza context).
6) Campioniamo residui sintetici e correggiamo Leaspy: x' = x_leaspy + r̂.
7) Valutiamo KS per feature e DCR (Distance to Closest Record).

Adatta i nomi colonne nei blocchi CONFIG.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

# SDV / SDMetrics
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer
try:
    from sdmetrics.single_table.privacy import DistanceToClosestRecord
except Exception:
    DistanceToClosestRecord = None


# ==============================
# Utils: tempo monotono per id
# ==============================
def ensure_monotone_per_id(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """Rende il time_col strettamente crescente per id (aggiunge jitter minimo se servisse)."""
    df = df.sort_values([id_col, time_col]).copy()

    def _fix(g: pd.DataFrame) -> pd.DataFrame:
        t = g[time_col].astype(float).to_numpy()
        for i in range(1, len(t)):
            if not (t[i] > t[i-1]):
                t[i] = np.nextafter(t[i-1], np.float64(np.inf))
        g[time_col] = t
        return g

    return df.groupby(id_col, group_keys=False).apply(_fix)


# ==============================
# Baseline & Matching KNN
# ==============================
def snapshot_baseline(
    df: pd.DataFrame, id_col: str, time_col: str, feature_cols: List[str]
) -> pd.DataFrame:
    """Estrae la prima riga (baseline) per id; usa solo le colonne utili al matching."""
    df = df.sort_values([id_col, time_col])
    base = df.groupby(id_col, as_index=False).first()
    keep = [id_col] + feature_cols
    return base[keep]


def knn_match_by_baseline_feats(
    df_real: pd.DataFrame,
    df_leaspy: pd.DataFrame,
    id_col: str,
    time_col: str,
    baseline_match_feats: List[str],
    k: int = 3
) -> Dict:
    """
    Ritorna un mapping: leaspy_id -> [real_id_1, ..., real_id_k]
    usando KNN su feature di baseline (senza usare variabili 'context').
    """
    real_base = snapshot_baseline(df_real, id_col, time_col, baseline_match_feats)
    leas_base = snapshot_baseline(df_leaspy, id_col, time_col, baseline_match_feats)

    scaler = StandardScaler().fit(real_base[baseline_match_feats])
    Xr = scaler.transform(real_base[baseline_match_feats])
    Xl = scaler.transform(leas_base[baseline_match_feats])

    nn = NearestNeighbors(
        n_neighbors=min(k, len(real_base)),
        metric='euclidean',
        algorithm='auto'
    ).fit(Xr)
    _d, idxs = nn.kneighbors(Xl)

    mapping = {
        leas_base[id_col].iloc[i]: real_base[id_col].iloc[idxs[i]].tolist()
        for i in range(len(leas_base))
    }
    return mapping


# ==============================
# Interpolazione al tempo target
# ==============================
def resample_to_times_numeric(
    g: pd.DataFrame,
    time_col: str,
    target_times: np.ndarray,
    cols: List[str]
) -> pd.DataFrame:
    """
    Interpola colonne numeriche 'cols' ai tempi target (1D interpolate).
    Per categoriche, usare strategie alternative (non gestite qui).
    """
    g = g.sort_values(time_col)
    out = pd.DataFrame({time_col: target_times})
    t_src = g[time_col].to_numpy().astype(float)
    for c in cols:
        v_src = g[c].to_numpy().astype(float)
        out[c] = np.interp(target_times.astype(float), t_src, v_src)
    return out


# ==============================
# Costruzione dataset residui
# ==============================
def build_residual_dataset_knn(
    df_real: pd.DataFrame,
    df_leaspy: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: List[str],
    baseline_match_feats: List[str],
    k: int = 3
) -> pd.DataFrame:
    """
    Per ciascun paziente Leaspy:
      - trova i k reali più simili (KNN sulle baseline_match_feats),
      - riallinea le sequenze reali dei vicini ai timepoint del Leaspy,
      - resid = media_tra_real_vicini_aligned - serie_leaspy.
    Ritorna un df con (id, time, feature_cols) = residui.
    """
    # Sanity check & ordinamento
    df_real = ensure_monotone_per_id(df_real, id_col, time_col)
    df_leaspy = ensure_monotone_per_id(df_leaspy, id_col, time_col)

    mapping = knn_match_by_baseline_feats(df_real, df_leaspy, id_col, time_col, baseline_match_feats, k=k)
    real_by_id = {pid: g for pid, g in df_real.groupby(id_col)}

    residual_rows = []
    for pid_leas, g_leas in df_leaspy.groupby(id_col):
        neighbors = mapping.get(pid_leas, [])
        if not neighbors:
            continue

        # tempi target = tempi del paziente Leaspy
        t = g_leas[time_col].to_numpy().astype(float)

        # allinea sequenze reali vicine ai tempi target e calcola media
        aligned_real = []
        for rid in neighbors:
            gr = real_by_id.get(rid)
            if gr is None or gr.empty:
                continue
            gr_i = resample_to_times_numeric(
                g=gr[[time_col] + feature_cols],
                time_col=time_col,
                target_times=t,
                cols=feature_cols
            )
            aligned_real.append(gr_i[feature_cols].to_numpy())

        if not aligned_real:
            continue

        real_mean = np.nanmean(np.stack(aligned_real, axis=0), axis=0)  # [len(t), n_features]
        leas_vals = g_leas[feature_cols].to_numpy().astype(float)
        resid = real_mean - leas_vals

        df_res = pd.DataFrame({id_col: pid_leas, time_col: t})
        for j, c in enumerate(feature_cols):
            df_res[c] = resid[:, j]
        residual_rows.append(df_res)

    residuals = pd.concat(residual_rows, ignore_index=True) if residual_rows else pd.DataFrame()
    return residuals


# ==============================
# Train PAR sui residui & correzione
# ==============================
def train_par_on_residuals_no_context(
    residuals: pd.DataFrame,
    id_col: str,
    time_col: str,
    epochs: int = 256,
    cuda: bool = True,
    verbose: bool = False
) -> PARSynthesizer:
    """
    Allena PARSynthesizer sui residui (senza context).
    """
    md = Metadata.detect_from_dataframe(residuals, table_name='residuals')
    md.set_sequence_key(id_col)
    md.set_sequence_index(time_col)
    md.validate()

    par = PARSynthesizer(
        md,
        enforce_min_max_values=True,
        enforce_rounding=True,
        epochs=epochs,
        cuda=cuda,
        verbose=verbose
    )
    par.fit(residuals)
    return par

def nearest_merge_by_id(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_col: str,
    time_col: str,
    tolerance: Optional[float] = None,
    direction: str = "nearest",
    suffixes=("", "_res")
) -> pd.DataFrame:
    """
    Esegue un merge_asof per ciascun id, abbinando i timepoint più vicini.
    - left/right DEVONO essere ordinati per time_col all'interno di ciascun id.
    - tolerance è espressa nelle stesse unità di time_col (es. anni o mesi).
      Se None, non limita la distanza massima.
    """
    out_parts = []
    # per efficienza: indicizzazioni
    right_by_id = {pid: g.sort_values(time_col) for pid, g in right.groupby(id_col)}
    for pid, gL in left.groupby(id_col):
        gR = right_by_id.get(pid)
        if gR is None or gR.empty:
            # nessun match: la parte destra rimarrà NaN
            gL = gL.copy()
        else:
            gL = gL.sort_values(time_col).copy()
            merged = pd.merge_asof(
                gL, gR.sort_values(time_col),
                on=time_col,
                direction=direction,
                tolerance=tolerance,
                suffixes=suffixes
            )
            gL = merged
        out_parts.append(gL)
    return pd.concat(out_parts, ignore_index=True)


def correct_leaspy_with_par_residuals(
    df_leaspy: pd.DataFrame,
    par: PARSynthesizer,
    id_col: str,
    time_col: str,
    feature_cols: List[str],
    tolerance: Optional[float] = None,   # <-- NUOVO: es. 0.25 anni (~3 mesi) o 3 se usi "mesi"
    direction: str = "nearest"           # "backward"/"forward" se preferisci vincolare la corrispondenza
) -> pd.DataFrame:
    """
    Campiona residui sintetici e corregge Leaspy: x' = x_leaspy + r̂,
    usando un MERGE_ASOF per ID e per tempo con tolleranza opzionale.
    """
    n_seq = df_leaspy[id_col].nunique()
    synth_res = par.sample(num_sequences=n_seq)

    # Teniamo solo colonne utili e rinominiamo eventuali doppioni
    keep = [id_col, time_col] + [c for c in feature_cols if c in synth_res.columns]
    synth_res = synth_res[keep].copy()

    # merge_asof per ID (tempo più vicino, entro tolerance se fornita)
    merged = nearest_merge_by_id(
        left=df_leaspy[[id_col, time_col] + feature_cols],
        right=synth_res,
        id_col=id_col,
        time_col=time_col,
        tolerance=tolerance,
        direction=direction,
        suffixes=("", "_res")
    )

    # somma dei residui (NaN -> 0)
    for c in feature_cols:
        res_col = f"{c}_res"
        if res_col not in merged.columns:
            merged[res_col] = 0.0
        merged[c] = merged[c] + merged[res_col].fillna(0.0)

    return merged[[id_col, time_col] + feature_cols].copy()


# ==============================
# Valutazione: KS & DCR
# ==============================
def ks_by_feature(real_df: pd.DataFrame, synth_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """KS two-sample per feature (più basso = distribuzioni più simili)."""
    scores = {}
    for c in feature_cols:
        r = real_df[c].dropna().to_numpy()
        s = synth_df[c].dropna().to_numpy()
        if len(r) > 10 and len(s) > 10:
            scores[c] = float(ks_2samp(r, s).statistic)
    return scores


def dcr_distance_to_closest_record(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, feature_cols: List[str]
) -> Optional[float]:
    """DCR (più alto = meglio per privacy). Restituisce None se SDMetrics non disponibile."""
    if DistanceToClosestRecord is None:
        return None
    try:
        return float(DistanceToClosestRecord.compute(
            real_data=real_df[feature_cols].dropna(),
            synthetic_data=synth_df[feature_cols].dropna()
        ))
    except Exception:
        return None


# ==============================
# MAIN di esempio (adatta ai tuoi dati)
# ==============================
if __name__ == "__main__":
    # ======== CONFIG: ADATTA AI TUOI NOMI COLONNE ========
    id_col = "patient_id"
    time_col = "age"  # oppure "months_since_baseline" (monotono crescente per id)

    # Feature per il MATCHING (baseline). Scegline poche ma significative a t0.
    baseline_match_feats: List[str] = [
        "MMSE", "ADAS13", "Hippocampus"  # <-- ESEMPI. Metti le tue.
    ]

    # Tutte le feature che vuoi correggere (biomarcatori longitudinali)
    feature_cols: List[str] = [
        # --- ESEMPI, sostituisci con le tue colonne ---
        "MMSE", "ADAS13", "RAVLT_immediate", "RAVLT_learning", "MOCA", "FAQ",
        "Entorhinal", "Fusiform", "Hippocampus", "ICV", "MidTemp", "Ventricles",
        "WholeBrain", "AV45", "FDG", "ABETA", "TAU", "PTAU"
    ]

    # ======== CARICAMENTO DATI ========
    # Sostituisci con i tuoi data loading:
    # df_real = pd.read_csv("real.csv")
    # df_leaspy = pd.read_csv("leaspy_synth.csv")
    raise_on_missing = True
    try:
        df_real
        df_leaspy
    except NameError:
        if raise_on_missing:
            raise RuntimeError(
                "Definisci df_real e df_leaspy (DataFrame) prima di eseguire lo script, "
                "oppure sostituisci i placeholder di caricamento."
            )

    # Filtra colonne necessarie
    cols_needed_real = list({id_col, time_col, *baseline_match_feats, *feature_cols})
    cols_needed_leas = list({id_col, time_col, *baseline_match_feats, *feature_cols})
    df_real = df_real[ [c for c in cols_needed_real if c in df_real.columns] ].copy()
    df_leaspy = df_leaspy[ [c for c in cols_needed_leas if c in df_leaspy.columns] ].copy()

    # Sanity: tempo monotono
    df_real = ensure_monotone_per_id(df_real, id_col, time_col)
    df_leaspy = ensure_monotone_per_id(df_leaspy, id_col, time_col)

    # ======== 1) RESIDUI via pseudo-pairing KNN ========
    residuals = build_residual_dataset_knn(
        df_real=df_real,
        df_leaspy=df_leaspy,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        baseline_match_feats=baseline_match_feats,
        k=3
    )
    if residuals.empty:
        raise RuntimeError("Residual dataset vuoto: verifica nomi colonne, baseline_match_feats e disponibilità dati.")

    # ======== 2) TRAIN PAR sui residui (senza context) ========
    par = train_par_on_residuals_no_context(
        residuals=residuals,
        id_col=id_col,
        time_col=time_col,
        epochs=256,
        cuda=True,
        verbose=False
    )

    # ======== 3) CORREZIONE Leaspy con residui sintetici ========
    df_corrected = correct_leaspy_with_par_residuals(
        df_leaspy=df_leaspy,
        par=par,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        tolerance=0.25,        # 0.25 anni ≈ 3 mesi
        direction="nearest"
    )

    # ======== 4) VALUTAZIONE (KS & DCR) ========
    ks_scores = ks_by_feature(df_real, df_corrected, feature_cols)
    dcr_score = dcr_distance_to_closest_record(df_real, df_corrected, feature_cols)

    # Report sintetico
    ks_tbl = pd.DataFrame({"feature": list(ks_scores.keys()), "KS": list(ks_scores.values())}).sort_values("KS")
    print("\n=== KS per feature (↓ meglio) ===")
    print(ks_tbl.to_string(index=False))

    print("\n=== DCR globale (↑ meglio) ===")
    print(dcr_score)
