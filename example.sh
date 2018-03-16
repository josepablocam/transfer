INPUT_DIR=data/
OUTPUT_DIR=data/extracted/

mkdir -p ${OUTPUT_DIR}

# get expressions
python3 extract.py ${INPUT_DIR}/input_data ${INPUT_DIR}/database.sqlite ${OUTPUT_DIR} --max_similarity_ratio 0.5
# abstract them out (naively)
python3 abstract.py ${OUTPUT_DIR}/df-exprs.pkl ${OUTPUT_DIR}/df-exprs-abstract.pkl
python3 abstract.py ${OUTPUT_DIR}/str-exprs.pkl ${OUTPUT_DIR}/str-exprs-abstract.pkl
# show groupings (naively)
python3 group.py ${OUTPUT_DIR}/df-exprs-abstract.pkl > ${OUTPUT_DIR}/grouped_dataframe_ops.txt
python3 group.py ${OUTPUT_DIR}/str-exprs-abstract.pkl > ${OUTPUT_DIR}/grouped_str_ops.txt


