echo "Running script $1"
(
    cd scripts/;
    mkdir -p generated/;
    echo 'Rewriting'
    python -m plpy.rewrite.expr_lifter $1.py generated/$1_lifted.py;
    echo 'Running and Tracing'
    ipython3 -m plpy.analyze.dynamic_tracer -- generated/$1_lifted.py generated/$1_tracer.pkl --loop_bound 2 --log $1.log
)
