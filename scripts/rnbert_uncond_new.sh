#!/usr/bin/env bash
set -Eeuo pipefail

HELPER="${HELPER:-scripts/helpers}"              #  folder where the helper scripts are stored, scripts/helpers by default
RESULTS="${RESULTS:-results}"                    # folder where intermediate results will be stored and where saved predictions are expected to be found, results by default
NAME="${NAME:-00trim}"                           # name of the folder with the predictions, e.g. baseline
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"              # folder where the final csv file with the metrics will be stored

mkdir -p "$OUTPUT_DIR"


set -e
set -x


 # works
python ${HELPER}/collate_predictions.py \
    metadata=/work/ui556004/data/rnbert/datasets/rnbert_abstract_data_raw/metadata_test.txt \
    predictions=${RESULTS}/saved_predictions/${NAME}/test/predictions \
    prediction_file_type=both \
    output_folder=${RESULTS}/collated/test/outputs_${NAME} \
    overwrite=True \
    error_if_exists=False \
    n_specials_to_ignore=0




# get per-salami-slice preds

python ${HELPER}/get_per_salami_slice_preds.py \
    column_types.inversion=float \
    metadata=${RESULTS}/test/outputs/metadata_test.txt \
    predictions=${RESULTS}/collated/test/outputs_${NAME}/predictions \
    dictionary_folder=${RESULTS}/saved_predictions/${NAME}/test \
    output_folder=${RESULTS}/test/salamied_${NAME} \
    concat_features='[[primary_alteration,primary_degree,secondary_alteration,secondary_degree],[key_pc,mode]]' \
    n_specials=0 \
    collated=True

#calculate metrics

bash ${HELPER}/musicbert_synced_metrics_concat_degree.sh \
    ${RESULTS}/test/salamied_${NAME} \
    ${OUTPUT_DIR}/output_synced_metrics_${NAME}.csv --uniform-steps


