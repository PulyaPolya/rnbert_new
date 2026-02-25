import os
from transformers import AutoConfig
import ast
import numpy as np
import sklearn
import pandas as pd
import sklearn.metrics
from music21 import roman, key
import re
from typing import Optional
from music21 import roman, key
from music21.analysis import harmonicFunction as hf
from tqdm import tqdm
import argparse

_PC_TO_TONIC = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


def _format_alt_for_music21(alt: str) -> str:
    """
    - 'x'  -> '##' (double sharp)
    - 'xb' -> 'bb' (double flat)
    """
    if alt == "x":
        return "##"
    if alt == "xb":
        return "bb"
    return alt


def _split_primary_secondary(ps_token: str):
    """
    ps_token examples:
      '_V_I', 'bII_I', '#VII_VI', '_xbVII', 'VIIbVI', '_x_x', etc.

    Returns (primary_str, secondary_str)
    where each is alteration+degree: 'V', 'bII', '#VII', 'xbVII', etc.
    """
    # Strip leading underscore(s) first
    s = ps_token.lstrip("_")
    if not s:
        return None, None

    parts = s.split("_")
    parts = [p for p in parts if p != ""]

    if len(parts) == 2:
        # typical '_V_I', '_V_V', etc.
        return parts[0], parts[1]
    elif len(parts) == 1:
        # packed form like 'VIIbVI', 'V#III', 'xbVII', etc.
        m = re.match(
            r"^(?P<alt1>b|#|x|xb)?(?P<deg1>I{1,3}|IV|V|VI|VII)"
            r"(?P<alt2>b|#|x|xb)?(?P<deg2>I{1,3}|IV|V|VI|VII)?$",
            parts[0],
        )
        if not m:
            # no structured match – treat everything as primary
            return parts[0], None

        alt1 = m.group("alt1") or ""
        deg1 = m.group("deg1")
        alt2 = m.group("alt2") or ""
        deg2 = m.group("deg2")

        primary = alt1 + deg1
        secondary = alt2 + deg2 if deg2 else None
        return primary, secondary
    else:
        return None, None


def _split_alt_degree(deg_token: Optional[str]):
    """
    deg_token like 'V', 'bII', '#VII', 'xVI', 'xbVII', or None.
    Returns (accidental, numeral) e.g. ('b', 'II'), ('', 'V').
    """
    if deg_token is None:
        return None, None

    m = re.match(r"^(?P<alt>b|#|x|xb)?(?P<deg>I{1,3}|IV|V|VI|VII)$", deg_token)
    if not m:
        # something exotic – return "no alt" and the raw token as degree
        return "", deg_token

    alt = m.group("alt") or ""
    deg = m.group("deg")
    return alt, deg


def _build_rn_main(primary_deg_full: str, quality: str, inversion: str) -> str:
    """
    Build the *main* Roman numeral (without /secondary) in music21 style,
    including quality & inversion.

    Special case: quality == 'aug6' → treat as Italian augmented 6th: 'It+6'.
    """
    # ---- special case: augmented sixth chords ---------------------------
    if quality == "aug6":
        # but as a baseline we map *all* aug6 to It+6.
        return "It+6"

    # ---- generic case ---------------------------------------------------
    alt, core = _split_alt_degree(primary_deg_full)
    if alt is None:
        return "N.C."

    alt_out = _format_alt_for_music21(alt)

    is_seventh = quality in {"Mm7", "M7", "m7", "o7", "ø7"}
    is_dim = quality in {"o", "o7"}
    is_half_dim = quality == "ø7"
    is_minor = quality in {"m", "m7"}

    # case (upper vs lower)
    if is_minor or is_dim or is_half_dim:
        core_rn = core.lower()
    else:
        core_rn = core.upper()

    base = alt_out + core_rn

    # add quality markers
    if quality in {"o", "o7"}:
        base += "o"
    elif quality == "ø7":
        base += "ø"
    elif quality == "+":
        base += "+"

    # inversion figures
    if is_seventh:
        inv_map7 = {"0.0": "7", "1.0": "65", "2.0": "43", "3.0": "42"}
        figure = inv_map7.get(inversion, "7")
        rn = base + figure
    else:
        inv_map3 = {"0.0": "", "1.0": "6", "2.0": "64"}
        figure = inv_map3.get(inversion, "")
        rn = base + figure

    return rn


# -------------------------------------------------------------------------
# main function
# here the predictions done by RNbert are a converted to match the ofrmat of music21 library

def chord_string_to_music21(chord_str: str) -> roman.RomanNumeral:
    """
    Convert RN encoding string, e.g.
      'Mm7_2.0_6.0M__V_I'
      'o7_3.0_0.0M_#VII_VI'
      'aug6_1.0_9.0M__xbVII'
    into a music21 RomanNumeral object.
    """
    # quality, inversion, key_pc_mode, the rest (rn-part)
    q, inv, key_pm, ps = chord_str.split("_", 3)

    # ---- key decoding ---------------------------------------------------
    pc_str, mode_char = key_pm[:-1], key_pm[-1]  # '9.0M' -> '9.0', 'M'
    pc = int(float(pc_str))
    tonic = _PC_TO_TONIC[pc]
    mode = "major" if mode_char == "M" else "minor"
    k = key.Key(tonic, mode)

    # ---- primary/secondary ----------------------------------------------
    primary_str, secondary_str = _split_primary_secondary(ps)

    # for aug6, primary_str might be 'xbVII' but we ignore it in rn symbol
    if primary_str is None:
        rn_main = "N.C."
    else:
        rn_main = _build_rn_main(primary_str, q, inv)

    # attach secondary if present
    if secondary_str:
        # also map its accidental for music21
        alt2, deg2 = _split_alt_degree(secondary_str)
        if deg2 is not None:
            sec = _format_alt_for_music21(alt2) + deg2
            rn_full = f"{rn_main}/{sec}"
        else:
            rn_full = rn_main
    else:
        rn_full = rn_main

    return roman.RomanNumeral(rn_full, k)


def parse_cell(x):
    if not isinstance(x, str):
        return x
    res = ast.literal_eval(x)
    #print(max(res))
    return(max(res))
from typing import Dict, List, Union
import random
def get_random_key(id2label_dict):
    # do not choose special symbols 
    keys = list(id2label_dict.keys())[4:]
    random_key =  random.choice(keys)
    return id2label_dict[random_key]
def _decode_ids_to_tokens(ids: np.ndarray, id2label: Dict[int, str]) -> List[str]:
    # Map each int id -> label string; unknown ints -> str(id)
    # in case there is some wrong key just choose the value randomly
    return [id2label.get(str(i), get_random_key(id2label)) for i in ids.tolist()]
mapping_dict = {"labels_0": "quality", "labels_1": "inversion", "labels_2": "key_pc_mode", "labels_3": "primary_alteration_primary_degree_secondary_alteration_secondary_degree"}
def split_rn(rn):
    rn_wo_degree, degree = rn.split("__")
    quality, inversion, key_pc_mode = rn_wo_degree.split("_")
    return quality, inversion, key_pc_mode, degree


def get_function_percentage(all_label_values, all_predicted_values, onlyHauptHarmonicFunction= False):
    total_size = len(all_label_values)
    not_same = 0
    same_function = 0
    not_real_rn = 0
    for i in tqdm(range(total_size)):
        try:
            true_rn = chord_string_to_music21(all_label_values[i])
            pred_rn = chord_string_to_music21(all_predicted_values[i])
            if true_rn != pred_rn:
                not_same += 1
                hf_true = hf.romanToFunction(true_rn, onlyHauptHarmonicFunction=onlyHauptHarmonicFunction)
                hf_pred = hf.romanToFunction(pred_rn, onlyHauptHarmonicFunction=onlyHauptHarmonicFunction) 
                if hf_true == hf_pred:
                    same_function += 1
        except Exception as e:
            not_real_rn += 1
            pass
    print(f" Total size is {total_size}, wrong answers in total: {not_same}, from them the same function: {round(same_function / not_same *100, 2)}%")
    print(f" Skipped {not_real_rn} entries since they could not be converted to rn")

def decode_rn (df):
    
    total_len = 0
    decoded_labels_column = []
    decoded_predictions_column = []
    all_label_values = []
    all_predicted_values = []
    for ind, row in df.iterrows():
        y_trues = [
            np.array(ast.literal_eval(row[f"labels_{i}"])) 
            for i in range(len(input_files))
        ]
        y_trues_decoded = [_decode_ids_to_tokens(y_true_enc, id2label_map[mapping_dict[f"labels_{i}"]])  for i, y_true_enc in enumerate(y_trues)]
        
        y_preds = [
            np.array(ast.literal_eval(row[f"predicted_{i}"]))
            for i in range(len(input_files))
        ]
        y_preds_decoded = [_decode_ids_to_tokens(y_pred_enc +4, id2label_map[mapping_dict[f"labels_{i}"]])  for i, y_pred_enc in enumerate(y_preds)]
        y_true = ["_".join(str(x) for x in xs) for xs in zip(*y_trues)]

        y_true_decoded = ["_".join(x for x  in xs) for xs in zip(*y_trues_decoded)]
        y_pred_decoded = ["_".join(x for x  in xs) for xs in zip(*y_preds_decoded)]
        decoded_labels_column.append(y_true_decoded)
        decoded_predictions_column.append(y_pred_decoded)
        y_pred = ["_".join(str(x + 4) for x in xs) for xs in zip(*y_preds)] 
        uniform_step_col = row["uniform_steps"]

        repeats =  np.array(ast.literal_eval(uniform_step_col))
        y_true = np.repeat(y_true, repeats)
        y_pred = np.repeat(y_pred, repeats)
        all_label_values += y_true_decoded
        all_predicted_values += y_pred_decoded
        assert len(y_true) == len(y_pred)

        unique_labels = sorted(set(y_true) | set(y_pred))
        total_len += len(y_true)
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)  
        df.at[ind, "accuracy"] = accuracy
    df["labels_decoded"] = decoded_labels_column
    df["preds_decoded"] = decoded_predictions_column
    return df, all_label_values, all_predicted_values
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the model checkpoint")
parser.add_argument("--root_dir", type=str, help="Path to the directory with decoded predictions" )
args = parser.parse_args()
config = AutoConfig.from_pretrained(args.path)
id2label_map = config.multitask_id2label
input_files = os.listdir(args.root_dir)
input_files = [os.path.join(args.root_dir, file) for file in input_files]
df = pd.read_csv(input_files[0])
paths_df_indices = df[["path", "indices"]]
df = df.drop(["indices"], axis=1) #  
for i, input_file in enumerate(input_files[1:], start=1):
    new_df = pd.read_csv(input_file).drop(["path", "indices"], axis=1)
    df = df.merge(
        new_df,
        left_index=True,
        right_index=True,
        suffixes=("", f"_{i}"),
        how="outer",
    )
df = df.rename({"predicted": "predicted_0", "labels": "labels_0"}, axis=1)
df, all_label_values, all_predicted_values  = decode_rn (df)
upper, lower = 0.95, 0.3
print(f"The worst pieces with accuracy < {lower} are:")
print(df[df["accuracy"] < lower]["path"].values)
print(f"The best pieces with accuracy >{upper} are:")
print(df[df["accuracy"] > upper]["path"].values)
get_function_percentage(all_label_values, all_predicted_values, onlyHauptHarmonicFunction= False)