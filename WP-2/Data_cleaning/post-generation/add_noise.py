import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict

from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer

# valutazione
from scipy.stats import ks_2samp
try:
    # SDMetrics ≥ 0.12
    from sdmetrics.single_table.privacy import DistanceToClosestRecord
except Exception:
    DistanceToClosestRecord = None

from sklearn.preprocessing import QuantileTransformer


# -----------------------------
# 1) Utilità: metadata & pulizia
# -----------------------------
def build_sequential_metadata(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    context_cols: Optional[List[str]] = None
) -> Metadata:
    """
    Crea/aggiorna Metadata sequenziale per SDV 1.x.
    """
    md = Metadata.detect_from_dataframe(df, table_name='longitudinal')
    md.set_sequence_key(column_name=id_col)
    md.set_sequence_index(column_name=time_col)
    if context_cols:
        md.set_context_columns(context_cols)
    md.validate()
    return md


def ensure_monotone_per_id(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """
    Rimuove non-monotonicità e tie su time_col per ogni id (aggiunge jitter minimo).
    """
    df = df.sort_values([id_col, time_col]).copy()
    # risolve duplicati su time entro paziente
    def _fix_group(g):
        t = g[time_col].to_numpy(dtype=float)
        # forza strettamente crescente con jitter molto piccolo
        for i in range(1, len(t)):
            if not (t[i] > t[i-1]):
                t[i] = np.nextafter(t[i-1], np.float64(np.inf))
        g[time_col] = t
        return g
    return df.groupby(id_col, group_keys=False).apply(_fix_group)


def align_real_and_leaspy(
    df_real: pd.DataFrame,
    df_leaspy: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: List[str],
    tolerance: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allinea (inner-join) reali e Leaspy su (id, time).
    Se 'tolerance' è specificata, fa un 'nearest merge' entro la tolleranza sul tempo
    (utile se i timepoint non coincidono esattamente).
    """
    real = df_real[[id_col, time_col] + feature_cols].copy()
    leas = df_leaspy[[id_col, time_col] + feature_cols].copy()

    real = ensure_monotone_per_id(real, id_col, time_col)
    leas = ensure_monotone_per_id(leas, id_col, time_col)

    if tolerance is None:
        merged = pd.merge(real, leas, on=[id_col, time_col], suffixes=("_real", "_leas"), how='inner')
    else:
        # nearest merge per id: per ciascun id, fai un merge_asof sul tempo
        merged_list = []
        for pid, g_real in real.groupby(id_col):
            g_leas = leas[leas[id_col] == pid]
            if g_leas.empty:
                continue
            g_real = g_real.sort_values(time_col)
            g_leas = g_leas.sort_values(time_col)
            m = pd.merge_asof(
                g_real, g_leas,
                on=time_col, direction='nearest', tolerance=tolerance,
                suffixes=("_real", "_leas")
            )
            m[id_col] = pid
            merged_list.append(m)
        merged = pd.concat(merged_list, ignore_index=True)

    # split di ritorno
    real_aligned = merged[[id_col, time_col] + [f"{c}_real" for c in feature_cols]].rename(
        columns={f"{c}_real": c for c in feature_cols}
    )
    leas_aligned = merged[[id_col, time_col] + [f"{c}_leas" for c in feature_cols]].rename(
        columns={f"{c}_leas": c for c in feature_cols}
    )
    return real_aligned, leas_aligned


# -----------------------------
# 2) Residual-PAR
# -----------------------------
def compute_residuals(
    real: pd.DataFrame,
    leas: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: List[str],
    context_cols: Optional[List[str]] = None,
    context_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calcola r_t = x_real_t - x_leaspy_t per feature.
    Se context_df è fornito, lo join-a per id come colonne (costanti nel tempo).
    """
    df = real[[id_col, time_col] + feature_cols].copy()
    for c in feature_cols:
        df[c] = real[c].values - leas[c].values

    if context_cols and context_df is not None:
        # sanity: tieni solo le context che esistono
        keep = [c for c in context_cols if c in context_df.columns]
        ctx = context_df[[id_col] + keep].drop_duplicates(id_col)
        df = df.merge(ctx, on=id_col, how='left')

    return df


def train_par_on_residuals(
    residuals: pd.DataFrame,
    id_col: str,
    time_col: str,
    context_cols: Optional[List[str]] = None,
    epochs: int = 128,
    cuda: bool = True,
    enforce_min_max: bool = True,
    enforce_rounding: bool = True,
    segment_size: Optional[int] = None,
    verbose: bool = False
) -> Tuple[Metadata, PARSynthesizer]:
    """
    Costruisce metadata e allena PAR sui residui.
    """
    md = build_sequential_metadata(residuals, id_col, time_col, context_cols)
    par = PARSynthesizer(
        md,
        enforce_min_max_values=enforce_min_max,
        enforce_rounding=enforce_rounding,
        epochs=epochs,
        cuda=cuda,
        verbose=verbose
    )
    # opzionale: segment_size non è argomento diretto; in SDV 1.x è tra le NN options (vedi docs).
    # Alcune versioni espongono 'segment_size' via kwargs nel costruttore. Se disponibile:
    if segment_size is not None and hasattr(par, "_model_kwargs"):
        par._model_kwargs['segment_size'] = segment_size  # fallback soft; ignora se non supportato

    par.fit(residuals)
    return md, par


def correct_leaspy_with_synthetic_residuals(
    leaspy_df: pd.DataFrame,
    par: PARSynthesizer,
    num_sequences: Optional[int],
    id_col: str,
    time_col: str,
    feature_cols: List[str],
    context_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Campiona residui sintetici con PAR e corregge Leaspy: x' = x_leaspy + r̂.
    Mantiene (id, time) delle sequenze sintetiche: si farà un merge dopo.
    """
    # Sampling: se vuoi una correzione 1:1 per ogni sequenza di Leaspy,
    # puoi campionare lo stesso numero di sequenze e poi riallineare per (id,time).
    if num_sequences is None:
        num_sequences = leaspy_df[id_col].nunique()

    synth_resid = par.sample(num_sequences=num_sequences)  # shape: [id_col, time_col, feature_cols(+ctx)]
    # Teniamo solo le colonne utili
    keep_cols = [id_col, time_col] + feature_cols
    synth_resid = synth_resid[[c for c in keep_cols if c in synth_resid.columns]].copy()

    # merge (id,time): se gli indici tempo non combaciano, usa merge_asof per id
    # Qui assumiamo combacino; se no, sostituisci con nearest merge per gruppo id.
    merged = pd.merge(
        leaspy_df[[id_col, time_col] + feature_cols],
        synth_resid, on=[id_col, time_col],
        suffixes=("", "_res"), how="left"
    )

    # se mancano residui (NaN), metti 0
    for c in feature_cols:
        if f"{c}_res" not in merged.columns:
            merged[f"{c}_res"] = 0.0
        merged[f"{c}_res"] = merged[f"{c}_res"].fillna(0.0)
        merged[c] = merged[c] + merged[f"{c}_res"]

    corrected = merged[[id_col, time_col] + feature_cols].copy()
    return corrected


# -----------------------------
# 3) Marginal correction (opz.)
# -----------------------------
def quantile_map_to_real(
    corrected_df: pd.DataFrame,
    real_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Applica un QuantileTransformer per mappare le marginali delle feature
    del dataset corretto su quelle del dataset reale (conserva l'ordine per feature).
    """
    out = corrected_df.copy()
    for c in feature_cols:
        real_vals = real_df[c].dropna().to_numpy().reshape(-1, 1)
        corr_vals = out[c].to_numpy().reshape(-1, 1)

        if len(np.unique(real_vals)) < 5:
            # troppi pochi valori distinti: salta mappatura
            continue

        qt_fit = QuantileTransformer(n_quantiles=min(1000, len(real_vals)), output_distribution='uniform')
        real_u = qt_fit.fit_transform(real_vals)  # U(0,1) che preserva il rank dei reali

        # Ora mappa i corretti sullo spazio U, poi rimappa ai quantili reali
        qt_apply = QuantileTransformer(n_quantiles=min(1000, len(corr_vals)), output_distribution='uniform')
        corr_u = qt_apply.fit_transform(corr_vals)

        # inverte usando l'ECDF dei reali: approx via percentili
        # Costruiamo griglia percentili per reali
        grid = np.linspace(0.0, 1.0, num=1001)
        real_percentiles = np.quantile(real_vals.flatten(), grid)
        # Interpola corr_u sulla griglia
        out[c] = np.interp(corr_u.flatten(), grid, real_percentiles)
    return out


# -----------------------------
# 4) Valutazione: KS & DCR
# -----------------------------
def ks_by_feature(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, float]:
    """
    KS two-sample per ogni feature (ignora NaN).
    """
    scores = {}
    for c in feature_cols:
        r = real_df[c].dropna().to_numpy()
        s = synth_df[c].dropna().to_numpy()
        if len(r) > 10 and len(s) > 10:
            scores[c] = ks_2samp(r, s).statistic
    return scores


def dcr_distance_to_closest_record(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: List[str]
) -> Optional[float]:
    """
    Distanza al record più vicino (più alto = meglio per privacy).
    Se SDMetrics non è disponibile, torna None.
    """
    if DistanceToClosestRecord is None:
        return None
    try:
        return float(DistanceToClosestRecord.compute(
            real_data=real_df[feature_cols].dropna(),
            synthetic_data=synth_df[feature_cols].dropna()
        ))
    except Exception:
        return None


# -----------------------------
# 5) Pipeline orchestrator
# -----------------------------
def residual_par_pipeline(
    df_real: pd.DataFrame,
    df_leaspy: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: List[str],
    context_cols: Optional[List[str]] = None,
    context_df: Optional[pd.DataFrame] = None,
    tolerance: Optional[float] = None,  # ad es. '30' se time_col è mesi e vuoi nearest entro ±30 giorni
    par_epochs: int = 128,
    use_cuda: bool = True,
    do_quantile_map: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float], Optional[float]]:
    """
    Esegue:
      1) allineamento real/leaspy,
      2) residual-PAR training,
      3) campionamento residui e correzione,
      4) (opz.) marginal correction via quantile mapping,
      5) metriche KS e DCR.
    Ritorna: df_corrected, ks_per_feature, dcr_value
    """
    # 1) allineamento
    real_al, leas_al = align_real_and_leaspy(
        df_real, df_leaspy, id_col, time_col, feature_cols, tolerance=tolerance
    )

    # 2) residui
    residuals = compute_residuals(real_al, leas_al, id_col, time_col, feature_cols, context_cols, context_df)

    # 3) train PAR su residui
    md, par = train_par_on_residuals(
        residuals=residuals,
        id_col=id_col,
        time_col=time_col,
        context_cols=context_cols,
        epochs=par_epochs,
        cuda=use_cuda,
        enforce_min_max=True,
        enforce_rounding=True,
        segment_size=None,
        verbose=False
    )

    # 4) correggi Leaspy con residui sintetici
    corrected = correct_leaspy_with_synthetic_residuals(
        leaspy_df=leas_al,
        par=par,
        num_sequences=leas_al[id_col].nunique(),
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        context_cols=context_cols
    )

    # opzionale: mappa le marginali sui reali per abbassare ulteriormente KS
    if do_quantile_map:
        corrected = quantile_map_to_real(corrected, real_al, feature_cols)

    # Valutazione (flatten per riga; qui usiamo tutte le righe, senza aggregare per paziente)
    ks_scores = ks_by_feature(real_al, corrected, feature_cols)
    dcr_score = dcr_distance_to_closest_record(real_al, corrected, feature_cols)

    return corrected, ks_scores, dcr_score


# -----------------------------
# ESEMPIO D’USO (adatta ai tuoi dati)
# -----------------------------
if __name__ == "__main__":
    # Esempio schematico (sostituisci con il tuo caricamento dati)
    # df_real = pd.read_csv("real.csv")
    # df_leaspy = pd.read_csv("leaspy_synth.csv")

    # Schema colonne (ADATTA!)
    id_col = "patient_id"
    time_col = "age"  # o "months_since_baseline"
    context_cols = ["baseline_age", "sex", "education", "APOE"]
    # Esempio placeholder feature_cols:
    feature_cols = [
        "MMSE", "ADAS13", "RAVLT_immediate", "RAVLT_learning", "MOCA", "FAQ",
        "Entorhinal", "Fusiform", "Hippocampus", "ICV", "MidTemp", "Ventricles",
        "WholeBrain", "AV45", "FDG", "ABETA", "TAU", "PTAU"
    ]

    # # Esegui pipeline:
    # corrected, ks_scores, dcr_score = residual_par_pipeline(
    #     df_real, df_leaspy,
    #     id_col=id_col,
    #     time_col=time_col,
    #     feature_cols=feature_cols,
    #     context_cols=context_cols,
    #     context_df=df_real[[id_col] + context_cols].drop_duplicates(id_col),
    #     tolerance=None,         # o un float per nearest merge
    #     par_epochs=256,         # più epoche = più qualità (ma più costo)
    #     use_cuda=True,
    #     do_quantile_map=True
    # )
    # print("KS per feature:", ks_scores)
    # print("DCR (↑ meglio per privacy):", dcr_score)
    pass
