reembed_vdf \
    --dir ms_marco_hi_mr_te_ta_ur_bn/ \
    --new_model_name embed-multilingual-v3.0 \
    --text_column query \
    --env_file_path "../vector-io/.env" \
    --batch_size 96 \
    --input_type clustering \
    --overwrite | tee output.txt