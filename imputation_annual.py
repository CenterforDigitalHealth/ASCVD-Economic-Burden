#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pycountry
except ImportError:  # pragma: no cover
    pycountry = None


DISEASE_LABEL_BY_TAG = {
    "IHD": "Ischemic heart disease",
    "IS": "Ischemic stroke",
    "PAD": "Lower extremity peripheral arterial disease",
    "ALL": "ASCVD",
}
TAG_BY_DISEASE_LABEL = {v: k for k, v in DISEASE_LABEL_BY_TAG.items()}

MANUAL_ISO3 = {
    "Micronesia (Federated States of)": "FSM",
    "Republic of Korea": "KOR",
    "Bolivia (Plurinational State of)": "BOL",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Iran (Islamic Republic of)": "IRN",
    "Palestine": "PSE",
    "Democratic Republic of the Congo": "COD",
    "United States Virgin Islands": "VIR",
}


def normalize_disease_option(value: str) -> str:
    token = str(value).strip().upper()
    if token not in {"ALL", "IHD", "IS", "PAD"}:
        raise ValueError("disease option must be one of: ALL, IHD, IS, PAD")
    return token


def get_disease_tags(disease_option: str) -> List[str]:
    if disease_option == "ALL":
        return ["IHD", "IS", "PAD", "ALL"]
    return [disease_option]


def first_existing_path(paths: List[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No file found among candidates: {paths}")


def load_csv_with_fallback(paths: List[str]) -> pd.DataFrame:
    path = first_existing_path(paths)
    return pd.read_csv(path)


def to_iso3(name: str) -> Optional[str]:
    if pd.isna(name):
        return None
    if name in MANUAL_ISO3:
        return MANUAL_ISO3[name]
    if pycountry is None:
        return None
    try:
        return pycountry.countries.lookup(str(name)).alpha_3
    except Exception:
        return None


def to_safe_tag(text: str) -> str:
    tag = "".join(ch if ch.isalnum() else "_" for ch in str(text).upper())
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag.strip("_")


def prepare_annual_for_tag(df_annual: pd.DataFrame, tag: str) -> pd.DataFrame:
    annual = df_annual.copy()

    required = {"Country Code", "scenario", "year", "GDP_loss"}
    missing = sorted(required - set(annual.columns))
    if missing:
        raise ValueError(f"annual_results_{tag}.csv missing columns: {missing}")

    if tag == "ALL":
        annual = (
            annual.groupby(["Country Code", "scenario", "year"], as_index=False)[["GDP_loss"]]
            .sum()
            .copy()
        )
        annual["disease"] = "ASCVD"
    else:
        annual["disease"] = annual.get("disease", DISEASE_LABEL_BY_TAG[tag])
        annual = annual[annual["disease"] == DISEASE_LABEL_BY_TAG[tag]].copy()

    annual = annual[["Country Code", "scenario", "year", "disease", "GDP_loss"]].copy()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    annual["GDP_loss"] = pd.to_numeric(annual["GDP_loss"], errors="coerce")
    return annual.dropna(subset=["year", "GDP_loss"]).assign(year=lambda x: x["year"].astype(int))


def load_inputs(
    disease_option: str,
    aggregate_dir: str,
    annual_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tags = get_disease_tags(disease_option)
    agg_parts = []
    annual_parts = []
    total_parts = []

    for tag in tags:
        agg_paths = [
            os.path.join(aggregate_dir, f"aggregate_results_imputed_{tag}.csv"),
            os.path.join("data", f"aggregate_results_imputed_{tag}.csv"),
            os.path.join("tmpresults", f"aggregate_results_imputed_{tag}.csv"),
        ]
        ann_paths = [
            os.path.join(annual_dir, f"annual_results_{tag}.csv"),
            os.path.join("data", f"annual_results_{tag}.csv"),
            os.path.join("tmpresults", f"annual_results_{tag}.csv"),
        ]

        agg_df = load_csv_with_fallback(agg_paths)
        ann_df = load_csv_with_fallback(ann_paths)
        ann_df = prepare_annual_for_tag(ann_df, tag)

        if "disease" in agg_df.columns and tag != "ALL":
            agg_df = agg_df[agg_df["disease"] == DISEASE_LABEL_BY_TAG[tag]].copy()
        if "disease" in agg_df.columns and tag == "ALL":
            agg_df = agg_df[agg_df["disease"] == "ASCVD"].copy()

        agg_parts.append(agg_df)
        annual_parts.append(ann_df)
        total_parts.append(
            ann_df.groupby(["Country Code", "scenario", "disease"], as_index=False)[["GDP_loss"]]
            .sum()
            .rename(columns={"GDP_loss": "annual_total"})
        )

    final_economic = pd.concat(agg_parts, ignore_index=True)
    annual = pd.concat(annual_parts, ignore_index=True)
    total_annual = pd.concat(total_parts, ignore_index=True)

    if "GDPloss" not in final_economic.columns:
        raise ValueError("aggregate file missing GDPloss column")
    final_economic["GDPloss"] = pd.to_numeric(final_economic["GDPloss"], errors="coerce") * 1e9
    return final_economic, annual, total_annual


def build_sdi_df(sdi_file: Optional[str]) -> pd.DataFrame:
    if sdi_file is None:
        sdi_file = first_existing_path(["data/SDI.csv", "data/GBD_SDI.csv"])

    sdi = pd.read_csv(sdi_file)
    if "year_id" in sdi.columns:
        sdi = sdi[sdi["year_id"] == 2023].copy()
    if "location_id" in sdi.columns:
        sdi = sdi.drop_duplicates(subset=["location_id"], keep="first")

    if "Country Code" not in sdi.columns:
        if "location_name" not in sdi.columns:
            raise ValueError("SDI file needs either 'Country Code' or 'location_name'")
        sdi["Country Code"] = sdi["location_name"].apply(to_iso3)
    else:
        sdi["Country Code"] = sdi["Country Code"].astype(str)

    value_col = "mean_value" if "mean_value" in sdi.columns else "SDI"
    if value_col not in sdi.columns:
        raise ValueError("SDI file missing SDI value column (mean_value or SDI)")

    sdi_df = sdi[["Country Code", value_col]].copy()
    sdi_df = sdi_df.rename(columns={value_col: "SDI"})
    sdi_df = sdi_df.dropna(subset=["Country Code", "SDI"]).drop_duplicates(subset=["Country Code"])
    sdi_df["SDI"] = pd.to_numeric(sdi_df["SDI"], errors="coerce")
    sdi_df = sdi_df.dropna(subset=["SDI"]).reset_index(drop=True)
    return sdi_df


def find_sdi_donors(target_sdi: float, donor_country_df: pd.DataFrame, k: int = 15, weighted: bool = False) -> pd.DataFrame:
    tmp = donor_country_df.copy()
    tmp = tmp.dropna(subset=["SDI"]).drop_duplicates(subset=["Country Code"])
    tmp["dist"] = (tmp["SDI"] - target_sdi).abs()
    tmp = tmp.sort_values(["dist", "Country Code"]).head(k).copy()
    if weighted:
        tmp["weight"] = 1 / (tmp["dist"] + 1e-8)
    else:
        tmp["weight"] = 1.0
    return tmp


def compute_metrics(compare_df: pd.DataFrame) -> Dict[str, float]:
    compare_df = compare_df.copy()
    error = compare_df["imputed_loss"] - compare_df["GDP_loss"]

    denom_sum = compare_df["GDP_loss"].sum()
    wape = np.nan if denom_sum == 0 else error.abs().sum() / denom_sum
    rmse = np.sqrt((error ** 2).mean()) if len(compare_df) > 0 else np.nan
    corr = compare_df["GDP_loss"].corr(compare_df["imputed_loss"]) if len(compare_df) >= 2 else np.nan

    if len(compare_df) > 0:
        eps = compare_df["GDP_loss"].quantile(0.05)
        mask = compare_df["GDP_loss"] > eps
        if mask.sum() > 0:
            mape_trimmed = (
                (compare_df.loc[mask, "imputed_loss"] - compare_df.loc[mask, "GDP_loss"]).abs()
                / compare_df.loc[mask, "GDP_loss"]
            ).mean()
        else:
            mape_trimmed = np.nan
    else:
        mape_trimmed = np.nan

    return {"MAPE_trimmed": mape_trimmed, "WAPE": wape, "RMSE": rmse, "corr": corr}


def estimate_annual_share_pattern_by_disease(
    test_country: str,
    disease: str,
    annual_clean: pd.DataFrame,
    final_economic_clean: pd.DataFrame,
    k: int = 15,
    weighted: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    annual_d = annual_clean[annual_clean["disease"] == disease].copy()
    final_d = final_economic_clean[final_economic_clean["disease"] == disease].copy()

    target_row = final_d[final_d["Country Code"] == test_country].copy()
    if target_row.empty:
        raise ValueError(f"[{disease}] {test_country}: not in final_economic")
    target_sdi = target_row["SDI"].dropna().iloc[0]

    donor_pool = annual_d.loc[annual_d["Country Code"] != test_country, "Country Code"].unique()
    donor_country_df = (
        final_d.loc[final_d["Country Code"].isin(donor_pool), ["Country Code", "SDI"]]
        .drop_duplicates(subset=["Country Code"])
        .dropna(subset=["SDI"])
        .copy()
    )
    if len(donor_country_df) == 0:
        raise ValueError(f"[{disease}] donor_country_df empty")

    donors = find_sdi_donors(target_sdi, donor_country_df, k=k, weighted=weighted)
    donor_codes = donors["Country Code"].tolist()
    donor_annual = annual_d.loc[
        annual_d["Country Code"].isin(donor_codes), ["Country Code", "year", "GDP_loss"]
    ].copy()
    if donor_annual.empty:
        raise ValueError(f"[{disease}] donor annual data empty")

    donor_annual["country_total"] = donor_annual.groupby("Country Code")["GDP_loss"].transform("sum")
    donor_annual = donor_annual[donor_annual["country_total"] != 0].copy()
    donor_annual["share"] = donor_annual["GDP_loss"] / donor_annual["country_total"]
    donor_annual = donor_annual.merge(donors[["Country Code", "weight"]], on="Country Code", how="left")

    tmp = donor_annual.copy()
    tmp["weighted_share"] = tmp["share"] * tmp["weight"]
    mean_share = (
        tmp.groupby("year")
        .agg(weighted_sum=("weighted_share", "sum"), weight_sum=("weight", "sum"))
        .reset_index()
    )
    mean_share["share"] = mean_share["weighted_sum"] / mean_share["weight_sum"]
    mean_share = mean_share[["year", "share"]]

    s = mean_share["share"].sum()
    if s <= 0:
        raise ValueError(f"[{disease}] share sum <= 0")
    mean_share["share"] = mean_share["share"] / s
    return mean_share, donors


def compute_validation_series_by_disease(
    test_country: str,
    disease: str,
    annual_clean: pd.DataFrame,
    final_economic_clean: pd.DataFrame,
    k: int = 15,
    weighted: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    annual_d = annual_clean[annual_clean["disease"] == disease].copy()
    final_d = final_economic_clean[final_economic_clean["disease"] == disease].copy()

    true_series = annual_d.loc[annual_d["Country Code"] == test_country, ["year", "GDP_loss"]].copy()
    if true_series.empty:
        raise ValueError(f"[{disease}] {test_country}: no observed annual")

    total_row = final_d.loc[
        (final_d["Country Code"] == test_country) & (final_d["scenario"] == "val")
    ].copy()
    if total_row.empty:
        raise ValueError(f"[{disease}] {test_country}: no val total")

    # final_economic['GDPloss'] has already been converted to annual GDP_loss unit.
    test_total = total_row["GDPloss"].iloc[0]

    mean_share, donors = estimate_annual_share_pattern_by_disease(
        test_country=test_country,
        disease=disease,
        annual_clean=annual_clean,
        final_economic_clean=final_economic_clean,
        k=k,
        weighted=weighted,
    )
    mean_share["imputed_loss"] = mean_share["share"] * test_total

    compare = (
        true_series.merge(mean_share[["year", "share", "imputed_loss"]], on="year", how="inner")
        .sort_values("year")
        .copy()
    )
    if compare.empty:
        raise ValueError(f"[{disease}] {test_country}: compare empty")
    return compare, donors


def tune_k_for_one_disease(
    disease: str,
    annual_clean: pd.DataFrame,
    final_economic_clean: pd.DataFrame,
    k_grid: Tuple[int, ...],
    weighted: bool = False,
    min_country_n: int = 3,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[int], Dict[str, object]]:
    annual_d = annual_clean[annual_clean["disease"] == disease].copy()
    final_d = final_economic_clean[final_economic_clean["disease"] == disease].copy()

    observed_countries = sorted(set(annual_d["Country Code"].unique()))
    available_val_countries = set(final_d.loc[final_d["scenario"] == "val", "Country Code"].unique())
    test_countries = [c for c in observed_countries if c in available_val_countries]

    if len(test_countries) < min_country_n:
        return None, None, None, {
            "disease": disease,
            "n_countries": len(test_countries),
            "best_k": np.nan,
            "status": "too_few_countries",
        }

    rows = []
    for k in k_grid:
        for country in test_countries:
            try:
                compare, _ = compute_validation_series_by_disease(
                    test_country=country,
                    disease=disease,
                    annual_clean=annual_clean,
                    final_economic_clean=final_economic_clean,
                    k=k,
                    weighted=weighted,
                )
                metrics = compute_metrics(compare)
                rows.append({"disease": disease, "country": country, "k": k, **metrics, "status": "ok"})
            except Exception as exc:  # pragma: no cover
                rows.append(
                    {
                        "disease": disease,
                        "country": country,
                        "k": k,
                        "MAPE_trimmed": np.nan,
                        "WAPE": np.nan,
                        "RMSE": np.nan,
                        "corr": np.nan,
                        "status": f"error: {str(exc)}",
                    }
                )

    results_df = pd.DataFrame(rows)
    summary = (
        results_df.groupby("k", as_index=False)[["MAPE_trimmed", "WAPE", "RMSE", "corr"]]
        .mean()
        .sort_values(["WAPE", "RMSE", "corr"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    if summary.empty or summary["WAPE"].isna().all():
        return results_df, summary, None, {
            "disease": disease,
            "n_countries": len(test_countries),
            "best_k": np.nan,
            "status": "all_failed",
        }

    best_k = int(summary.iloc[0]["k"])
    meta = {
        "disease": disease,
        "n_countries": len(test_countries),
        "best_k": best_k,
        "best_WAPE": summary.iloc[0]["WAPE"],
        "best_RMSE": summary.iloc[0]["RMSE"],
        "best_corr": summary.iloc[0]["corr"],
        "status": "ok",
    }
    return results_df, summary, best_k, meta


def impute_missing_country_all_scenarios_by_disease(
    test_country: str,
    disease: str,
    annual_clean: pd.DataFrame,
    final_economic_clean: pd.DataFrame,
    k: int,
    weighted: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mean_share, donors = estimate_annual_share_pattern_by_disease(
        test_country=test_country,
        disease=disease,
        annual_clean=annual_clean,
        final_economic_clean=final_economic_clean,
        k=k,
        weighted=weighted,
    )

    final_d = final_economic_clean[final_economic_clean["disease"] == disease].copy()
    target_rows = final_d.loc[
        final_d["Country Code"] == test_country,
        ["Country Code", "disease", "scenario", "GDPloss", "SDI"],
    ].copy()
    if target_rows.empty:
        raise ValueError(f"[{disease}] {test_country}: target rows missing")

    out = []
    for _, row in target_rows.iterrows():
        total = row["GDPloss"]
        tmp = mean_share.copy()
        tmp["GDP_loss"] = tmp["share"] * total
        tmp["Country Code"] = test_country
        tmp["disease"] = disease
        tmp["scenario"] = row["scenario"]
        tmp["SDI"] = row["SDI"]
        out.append(tmp[["Country Code", "disease", "year", "scenario", "GDP_loss", "share", "SDI"]])

    out_df = pd.concat(out, ignore_index=True)
    return out_df, donors


def run_disease_specific_imputation_pipeline(
    annual_clean: pd.DataFrame,
    final_economic_clean: pd.DataFrame,
    k_grid: Tuple[int, ...] = (5, 10, 15, 20, 30),
    weighted: bool = False,
    min_country_n: int = 3,
) -> Dict[str, object]:
    diseases = sorted(final_economic_clean["disease"].dropna().unique())

    k_meta_list = []
    validation_detail_list = []
    validation_summary_list = []
    imputed_list = []

    for disease in diseases:
        print("\n==============================")
        print(f"Disease: {disease}")
        print("==============================")

        annual_d = annual_clean[annual_clean["disease"] == disease].copy()
        final_d = final_economic_clean[final_economic_clean["disease"] == disease].copy()

        annual_iso_d = set(annual_d["Country Code"].unique())
        donor_countries_d = sorted(annual_iso_d)
        missing_countries_d = sorted(set(final_d["Country Code"].unique()) - set(donor_countries_d))

        print(f"Observed annual countries: {len(donor_countries_d)}")
        print(f"Missing annual countries:  {len(missing_countries_d)}")

        val_detail_df, val_summary_df, best_k, meta = tune_k_for_one_disease(
            disease=disease,
            annual_clean=annual_clean,
            final_economic_clean=final_economic_clean,
            k_grid=k_grid,
            weighted=weighted,
            min_country_n=min_country_n,
        )
        k_meta_list.append(meta)

        if val_detail_df is not None:
            validation_detail_list.append(val_detail_df)
        if val_summary_df is not None:
            tmp = val_summary_df.copy()
            tmp["disease"] = disease
            validation_summary_list.append(tmp)

        print(f"Best k for {disease}: {best_k} | status = {meta['status']}")
        if best_k is None:
            print(f"Skip imputation for {disease} because best_k is None")
            continue

        disease_imputed = []
        for country in missing_countries_d:
            try:
                imp_df, _ = impute_missing_country_all_scenarios_by_disease(
                    test_country=country,
                    disease=disease,
                    annual_clean=annual_clean,
                    final_economic_clean=final_economic_clean,
                    k=best_k,
                    weighted=weighted,
                )
                disease_imputed.append(imp_df)
            except Exception as exc:  # pragma: no cover
                print(f"Failed imputation | disease={disease} | country={country} | {exc}")

        if len(disease_imputed) > 0:
            disease_imputed_df = pd.concat(disease_imputed, ignore_index=True)
            imputed_list.append(disease_imputed_df)

    k_summary_df = pd.DataFrame(k_meta_list) if len(k_meta_list) > 0 else pd.DataFrame()
    validation_detail_df = (
        pd.concat(validation_detail_list, ignore_index=True) if len(validation_detail_list) > 0 else pd.DataFrame()
    )
    validation_summary_by_disease = (
        pd.concat(validation_summary_list, ignore_index=True)
        if len(validation_summary_list) > 0
        else pd.DataFrame()
    )
    imputed_annual_df = (
        pd.concat(imputed_list, ignore_index=True)
        if len(imputed_list) > 0
        else pd.DataFrame(columns=["Country Code", "disease", "year", "scenario", "GDP_loss", "share", "SDI"])
    )

    annual_observed_val_df = annual_clean[
        ["Country Code", "disease", "year", "GDP_loss", "scenario"]
    ].copy()

    annual_full_val_df = (
        pd.concat(
            [
                annual_observed_val_df,
                imputed_annual_df[imputed_annual_df["scenario"] == "val"][
                    ["Country Code", "disease", "year", "GDP_loss", "scenario"]
                ],
            ],
            ignore_index=True,
        )
        .sort_values(["disease", "Country Code", "year"])
        .reset_index(drop=True)
    )

    annual_full_all_scenarios_df = (
        pd.concat(
            [
                annual_observed_val_df[["Country Code", "disease", "year", "scenario", "GDP_loss"]],
                imputed_annual_df[["Country Code", "disease", "year", "scenario", "GDP_loss"]],
            ],
            ignore_index=True,
        )
        .sort_values(["disease", "Country Code", "scenario", "year"])
        .reset_index(drop=True)
    )

    dup_val = (
        annual_full_val_df.groupby(["Country Code", "disease", "year", "scenario"])
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    dup_all = (
        annual_full_all_scenarios_df.groupby(["Country Code", "disease", "year", "scenario"])
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )

    print("\n====================================")
    print("Pipeline completed")
    print("====================================")
    print("k_summary_df shape:", k_summary_df.shape)
    print("validation_detail_df shape:", validation_detail_df.shape)
    print("validation_summary_by_disease shape:", validation_summary_by_disease.shape)
    print("imputed_annual_df shape:", imputed_annual_df.shape)
    print("annual_full_val_df shape:", annual_full_val_df.shape)
    print("annual_full_all_scenarios_df shape:", annual_full_all_scenarios_df.shape)
    print("dup_val rows:", len(dup_val))
    print("dup_all rows:", len(dup_all))

    return {
        "k_summary_df": k_summary_df,
        "validation_detail_df": validation_detail_df,
        "validation_summary_by_disease": validation_summary_by_disease,
        "imputed_annual_df": imputed_annual_df,
        "annual_observed_val_df": annual_observed_val_df,
        "annual_full_val_df": annual_full_val_df,
        "annual_full_all_scenarios_df": annual_full_all_scenarios_df,
        "dup_val": dup_val,
        "dup_all": dup_all,
    }


def aggregate_annual_values(
    annual_full_all_scenarios_df: pd.DataFrame,
    mode: str,
    country_meta_file: str,
) -> pd.DataFrame:
    mode = mode.lower()
    if mode == "none":
        return pd.DataFrame()

    parts = []

    if mode in {"world", "both"}:
        world_df = (
            annual_full_all_scenarios_df.groupby(["disease", "scenario", "year"], as_index=False)[["GDP_loss"]]
            .sum()
            .copy()
        )
        world_df.insert(0, "aggregate_level", "World")
        world_df.insert(1, "aggregate_name", "World")
        parts.append(world_df)

    if mode in {"region", "both"}:
        meta = pd.read_csv(country_meta_file)
        if "Country Code" not in meta.columns or "Region" not in meta.columns:
            raise ValueError(f"{country_meta_file} must contain 'Country Code' and 'Region'")
        region_map = meta[["Country Code", "Region"]].drop_duplicates(subset=["Country Code"])
        merged = annual_full_all_scenarios_df.merge(region_map, on="Country Code", how="left")
        region_df = (
            merged.dropna(subset=["Region"])
            .groupby(["Region", "disease", "scenario", "year"], as_index=False)[["GDP_loss"]]
            .sum()
            .rename(columns={"Region": "aggregate_name"})
        )
        region_df.insert(0, "aggregate_level", "Region")
        parts.append(region_df)

    if len(parts) == 0:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def parse_k_grid(value: str) -> Tuple[int, ...]:
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if token == "":
            continue
        items.append(int(token))
    if len(items) == 0:
        raise ValueError("k-grid must include at least one integer")
    return tuple(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annual imputation pipeline equivalent to annual.ipynb with CLI options."
    )
    parser.add_argument(
        "--disease",
        type=str,
        default="ALL",
        help="Disease selector: ALL, IHD, IS, PAD",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="none",
        choices=["none", "world", "region", "both"],
        help="Aggregate annual GDP_loss by world/region.",
    )
    parser.add_argument(
        "--aggregate-dir",
        type=str,
        default="results",
        help="Directory for aggregate_results_imputed_*.csv",
    )
    parser.add_argument(
        "--annual-dir",
        type=str,
        default="results",
        help="Directory for annual_results_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for annual imputation CSV outputs.",
    )
    parser.add_argument(
        "--aggregate-output",
        type=str,
        default=None,
        help="Output csv path for world/region aggregate result.",
    )
    parser.add_argument("--sdi-file", type=str, default=None, help="SDI file path (default auto-detect).")
    parser.add_argument(
        "--country-meta-file",
        type=str,
        default="data/countrycode.csv",
        help="Country metadata file with Region column.",
    )
    parser.add_argument("--k-grid", type=str, default="5,10,15,20,30", help="Comma-separated K candidates.")
    parser.add_argument("--weighted", action="store_true", help="Use SDI-distance weighted donors.")
    parser.add_argument("--min-country-n", type=int, default=3, help="Minimum countries for per-disease validation.")
    args = parser.parse_args()

    disease_option = normalize_disease_option(args.disease)
    k_grid = parse_k_grid(args.k_grid)

    if args.aggregate_output is None:
        args.aggregate_output = f"results/annual_imputation_aggregate_{disease_option}.csv"

    os.makedirs(args.output_dir, exist_ok=True)

    final_economic, annual, total_annual = load_inputs(
        disease_option=disease_option,
        aggregate_dir=args.aggregate_dir,
        annual_dir=args.annual_dir,
    )

    total_check = pd.merge(
        final_economic, total_annual, on=["Country Code", "scenario", "disease"], how="left"
    )
    total_check["diff"] = total_check["annual_total"] - total_check["GDPloss"]
    total_check["rel_diff"] = total_check["diff"] / total_check["GDPloss"]
    total_check["abs_diff"] = total_check["diff"].abs()
    bad = total_check[total_check["rel_diff"].abs() > 0.01]
    print(f"Countries with >1% difference: {len(bad)}")
    if len(bad) > 0:
        cols = ["Country Code", "scenario", "disease", "GDPloss", "annual_total", "diff", "rel_diff"]
        print(bad.sort_values("abs_diff", ascending=False)[cols].head(30))

    annual_df = annual[annual["Country Code"] != "ROU"].copy()
    sdi_df = build_sdi_df(args.sdi_file)
    final_economic = final_economic.merge(sdi_df, on=["Country Code"], how="left")

    annual_clean = annual_df.dropna(subset=["Country Code", "year", "GDP_loss", "disease"]).copy()
    annual_clean["year"] = annual_clean["year"].astype(int)

    final_economic_clean = final_economic.dropna(
        subset=["Country Code", "disease", "scenario", "GDPloss", "SDI"]
    ).copy()
    final_economic_clean["scenario"] = final_economic_clean["scenario"].astype(str)
    final_economic_clean = final_economic_clean[
        final_economic_clean["scenario"].isin(["val", "lower", "upper"])
    ].copy()

    results = run_disease_specific_imputation_pipeline(
        annual_clean=annual_clean,
        final_economic_clean=final_economic_clean,
        k_grid=k_grid,
        weighted=args.weighted,
        min_country_n=args.min_country_n,
    )

    annual_full_all_scenarios_df = results["annual_full_all_scenarios_df"]
    disease_names = sorted(annual_full_all_scenarios_df["disease"].dropna().astype(str).unique())
    for disease_name in disease_names:
        disease_tag = TAG_BY_DISEASE_LABEL.get(disease_name, to_safe_tag(disease_name))
        out_csv = os.path.join(args.output_dir, f"annual_imputation_{disease_tag}.csv")
        out_df = annual_full_all_scenarios_df[annual_full_all_scenarios_df["disease"] == disease_name].copy()
        out_df.to_csv(out_csv, index=False)
        print(f"Saved annual imputation csv: {out_csv}")

    aggregated_df = aggregate_annual_values(
        annual_full_all_scenarios_df=annual_full_all_scenarios_df,
        mode=args.aggregate,
        country_meta_file=args.country_meta_file,
    )
    if len(aggregated_df) > 0:
        os.makedirs(os.path.dirname(args.aggregate_output) or ".", exist_ok=True)
        aggregated_df.to_csv(args.aggregate_output, index=False)
        print(f"Saved aggregated annual values: {args.aggregate_output}")


if __name__ == "__main__":
    main()
