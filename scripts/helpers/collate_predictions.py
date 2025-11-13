import argparse
import glob
import logging
import os
import shutil
from ast import literal_eval
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Literal
import sys

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_df.script_helpers import read_config_oc

import re
import math

def _parse_index_field(s) -> list[int]:
    # Handle already-parsed lists
 
   
    res = [int(x[1:][:-1]) for x in re.findall(r'\(\d+\)', s)]
    # Extract all integers; works for "[np.int64(2), ...]", "[1, 2]", "array([1,2])", "range(…)", etc.
    return res

# "midpoint": take the left predictions up to the midpoint of the overlap, then take
#   the right predictions
# "left": take the left predictions for the whole overlap
# "right": take the right predictions for the whole overlap
# "average": (only implemented for .h5 logits) take the mean of left and right
# "weighted_average": (only implemented for .h5 logits) take a weighted average,
#   linearly interpolated so that at the left boundary the left weight is 1.0 and at the
#   right boundary it is 0.0
OverlapStrategy = Literal["midpoint", "left", "right", "average", "weighted_average"]

PredictionFileType = Literal["h5", "txt", "both"]

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    # metadata: path to metadata csv containing at least the following columns:
    # csv_path, df_indices. Rows should be in one-to-one correspondance with
    # predictions.
    metadata: str
    # predictions: path to a folder with either an .h5 file or a .txt file containing
    # predicted tokens. Rows should be in one-to-one correspondance with metadata.
    predictions: str
    output_folder: str

    # regex to filter score ids
    feature_names: list[str] = field(default_factory=lambda: [])

    # column_types: dict[str, str] = field(default_factory=lambda: {})
    debug: bool = False
    h5_overlaps: OverlapStrategy = "weighted_average"
    txt_overlaps: OverlapStrategy = "midpoint"
    prediction_file_type: PredictionFileType = "txt"
    # When predicting tokens we need to subtract the number of specials
    # TODO: (Malcolm 2024-01-08) implement or remove?
    # data_has_start_and_stop_tokens: bool = False
    overwrite: bool = False
    error_if_exists: bool = True
    n_specials_to_ignore: int = 0
    features_to_skip: list[str] = field(default_factory=lambda: [])
    subfolder_name: str = "predictions"

def _load_yaml(path: Path) -> dict:
    raw = path.read_text()
    if path.suffix.lower() in (".yml", ".yaml"):
        data = yaml.safe_load(raw) or {}
    return data

def load_config(path: str | os.PathLike) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = _load_yaml(p)
    # allow hyphenated keys in file
    data = {k.replace("-", "_"): v for k, v in data.items()}
    # validate keys
    valid = set(Config.__annotations__.keys())
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    args, remaining = parser.parse_known_args()
    return args, remaining


def merge_token_predictions_and_indices(
    predictions: list[str], indices: list[str], config: Config
):
    #int_indices: list[list[int]] = [literal_eval(i) for i in indices]
    # extract integer values from np.64(i) format
    int_indices: list[list[int]] = [_parse_index_field(i) for i in indices]
    split_predictions: list[list[str]] = [p.strip().split() for p in predictions]
    # The assumption is that the segmentation of the indices is straightforward in
    #   that they are in the same order in each segment.

    out_indices = []
    out_predictions = []
    for indxs, preds in zip(int_indices, split_predictions, strict=True):
        if out_indices:
            left_overlap_i = 0
            for left_overlap_i, indx in enumerate(out_indices):
                if indx == indxs[0]:
                    break
            right_overlap_i = len(out_indices) - left_overlap_i

            left_overlap_indxs = out_indices[left_overlap_i:]
            right_overlap_indxs = indxs[:right_overlap_i]
            assert left_overlap_indxs == right_overlap_indxs
            if not left_overlap_indxs:
                continue
            if config.txt_overlaps == "midpoint":
                midpoint = right_overlap_i // 2
                out_indices = out_indices[: left_overlap_i + midpoint]
                out_indices.extend(indxs[midpoint:])
                out_predictions = out_predictions[: left_overlap_i + midpoint]
                out_predictions.extend(preds[midpoint:])
            elif config.txt_overlaps == "right":
                out_indices = out_indices[:left_overlap_i]
                out_indices.extend(indxs)
                out_predictions = out_predictions[:left_overlap_i]
                out_predictions.extend(preds)

            elif config.txt_overlaps == "left":
                out_indices.extend(indxs[right_overlap_i:])
                out_predictions.extend(preds[right_overlap_i:])
            else:
                raise ValueError(
                    "Only 'midpoint', 'right', and 'left' strategies are "
                    "implemented for token predictions"
                )

        else:
            out_indices.extend(indxs)
            out_predictions.extend(preds)

    return out_predictions, out_indices


def merge_logits_and_indices(
    logits_list: list[np.ndarray], indices: list[str], config: Config
):
    int_indices: list[list[int]] = [_parse_index_field(i) for i in indices]
    # The assumption is that the segmentation of the indices is straightforward in
    #   that they are in the same order in each segment.

    out_indices = []

    # We have to create an array and update it at every iteration because otherwise
    #   we run into problems when the array is shorter than the overlap length
    out_predictions = np.array([])

    for indxs, logits in zip(int_indices, logits_list, strict=True):
        # If logits are padded, we need to crop them
        logits = logits[: len(indxs)]

        # Remove specials
        logits = logits[:, config.n_specials_to_ignore :]

        if out_indices:
            left_overlap_i = 0
            for left_overlap_i, indx in enumerate(out_indices):
                if indx == indxs[0]:
                    break
            right_overlap_i = len(out_indices) - left_overlap_i

            left_overlap_indxs = out_indices[left_overlap_i:]
            right_overlap_indxs = indxs[:right_overlap_i]
            assert left_overlap_indxs == right_overlap_indxs
            if not left_overlap_indxs:
                continue
            out_indices.extend(indxs[right_overlap_i:])

            if config.h5_overlaps == "midpoint":
                midpoint = right_overlap_i // 2
                out_predictions = np.concatenate(
                    [out_predictions[: left_overlap_i + midpoint], logits[midpoint:]],
                    axis=0,
                )
            elif config.h5_overlaps == "right":
                out_predictions = np.concatenate(
                    [out_predictions[:left_overlap_i], logits],
                    axis=0,
                )
            elif config.h5_overlaps == "left":
                out_predictions = np.concatenate(
                    [out_predictions, logits[right_overlap_i:]],
                    axis=0,
                )
            elif config.h5_overlaps == "average":
                overlap = (
                    out_predictions[left_overlap_i:] + logits[:right_overlap_i]
                ) / 2.0
                out_predictions = np.concatenate(
                    [
                        out_predictions[:left_overlap_i],
                        overlap,
                        logits[right_overlap_i:],
                    ],
                    axis=0,
                )
            elif config.h5_overlaps == "weighted_average":
                left = (
                    out_predictions[left_overlap_i:]
                    * np.linspace(1.0, 0.0, right_overlap_i)[:, None]
                )
                right = (
                    logits[:right_overlap_i]
                    * np.linspace(0.0, 1.0, right_overlap_i)[:, None]
                )
                overlap = left + right
                out_predictions = np.concatenate(
                    [out_predictions[:left_overlap_i], overlap, logits[right_overlap_i:]],
                    axis=0,
                )
            else:
                raise ValueError

        else:
            out_indices.extend(indxs)
            out_predictions = logits

    assert len(out_predictions) == len(out_indices)

    return out_predictions, out_indices


def handle_metadata(metadata_rows, reference_df: pd.DataFrame | None, config: Config):
    out_metadata_df = pd.DataFrame(metadata_rows)
    if reference_df is None:
        df_path = os.path.join(config.output_folder, os.path.basename(config.metadata))
        out_metadata_df.to_csv(df_path)
        print(f"Wrote {df_path}")
        return out_metadata_df
    else:
        assert out_metadata_df.equals(reference_df)
        return reference_df


def main(config):
    #args, remaining = parse_args()
    #config = read_config_oc(args.config_file, Config)
    metadata_df = pd.read_csv(config.metadata, index_col=0).reset_index(drop=True)
    print(metadata_df["df_indices"].head(1).tolist())

    # (Malcolm 2024-01-08) There's no reason to be predicting on augmented
    #   data, which might lead to headaches.
    if "transpose" in metadata_df.columns:
        transposed = metadata_df["transpose"] != 0
        if transposed.any():
            LOGGER.warning(f"Found transposed items, removing")
            metadata_df = metadata_df[~transposed]

    if "scaled_by" in metadata_df.columns:
        scaled = metadata_df["scaled_by"] != 1.0
        if scaled.any():
            LOGGER.warning(f"Found scaled items, removing")
            metadata_df = metadata_df[~scaled]

    if os.path.exists(config.output_folder):
        if config.overwrite:
            shutil.rmtree(config.output_folder)
        elif config.error_if_exists:
            raise ValueError(f"Output folder {config.output_folder} already exists")
    os.makedirs(config.output_folder, exist_ok=True)

    unique_scores = metadata_df.score_id.unique()
    # The assumption is that the collated metadata file should be the same
    #   for all predictions (in the multi-task case).
    assert config.prediction_file_type in {"txt", "h5", "both"}

    reference_out_metadata_df = None
    if config.prediction_file_type in {"txt", "both"}:
        predictions_paths = glob.glob(os.path.join(config.predictions, "*.txt"))

        for predictions_path in predictions_paths:
            if os.path.basename(predictions_path)[:-4] in config.features_to_skip:
                print(f"Skipping {predictions_path}")
                continue
            out_preds_path = os.path.join(
                config.output_folder,
                config.subfolder_name,
                os.path.basename(predictions_path),
            )
            if not config.overwrite and os.path.exists(out_preds_path):
                print(f"{out_preds_path} exists, skipping...")
                continue
            print(f"Handling {predictions_path}")

            metadata_rows = []
            out_predictions = []
            with open(predictions_path) as inf:
                predictions_list = inf.readlines()

            for score in tqdm(unique_scores):
                score_rows = metadata_df[metadata_df.score_id == score]

                score_predictions_list = [predictions_list[i] for i in score_rows.index]
                (
                    merged_predictions,
                    merged_indices,
                ) = merge_token_predictions_and_indices(
                    score_predictions_list, score_rows.df_indices.tolist(), config
                )
                metadata_row = score_rows.iloc[0].copy()
                metadata_row["df_indices"] = merged_indices
                metadata_rows.append(metadata_row)
                out_predictions.append(merged_predictions)
                if config.debug:
                    break

            reference_out_metadata_df = handle_metadata(
                metadata_rows, reference_out_metadata_df, config
            )

            os.makedirs(os.path.dirname(out_preds_path), exist_ok=True)
            with open(out_preds_path, "w") as outf:
                for tokens in out_predictions:
                    outf.write(" ".join(tokens))
                    outf.write("\n")
            print(f"Wrote {out_preds_path}")
            if config.debug:
                break

    if config.prediction_file_type in {"h5", "both"}:
        predictions_paths = glob.glob(os.path.join(config.predictions, "*.h5"))

        for predictions_path in predictions_paths:
            if os.path.basename(predictions_path)[:-3] in config.features_to_skip:
                print(f"Skipping {predictions_path}")
                continue
            out_preds_path = os.path.join(
                config.output_folder,
                config.subfolder_name,
                os.path.basename(predictions_path),
            )
            if not config.overwrite and os.path.exists(out_preds_path):
                print(f"{out_preds_path} exists, skipping...")
                continue
            print(f"Handling {predictions_path}")

            metadata_rows = []
            out_predictions = []
            h5file = h5py.File(predictions_path, mode="r")

            os.makedirs(os.path.dirname(out_preds_path), exist_ok=True)
            h5outf = h5py.File(out_preds_path, "w")
            for score_i, score in enumerate(tqdm(unique_scores)):
                score_rows = metadata_df[metadata_df.score_id == score]

                score_predictions: list[np.ndarray] = [
                    (h5file[f"logits_{i}"])[:] for i in score_rows.index  # type:ignore
                ]

                (
                    merged_predictions,
                    merged_indices,
                ) = merge_logits_and_indices(
                    score_predictions, score_rows.df_indices.tolist(), config
                )
                metadata_row = score_rows.iloc[0].copy()
                metadata_row["df_indices"] = merged_indices
                metadata_rows.append(metadata_row)
                h5outf.create_dataset(f"logits_{score_i}", data=merged_predictions)
                out_predictions.append(merged_predictions)
                if config.debug:
                    break
            h5outf.close()
            reference_out_metadata_df = handle_metadata(
                metadata_rows, reference_out_metadata_df, config
            )

            print(f"Wrote {out_preds_path}")
            if config.debug:
                break


if __name__ == "__main__":
    cfg =load_config("collate_params.yaml")
   # args = parser.parse_args()

    main(cfg)
