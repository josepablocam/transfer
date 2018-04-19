echo "Running script $1"
(
    cd scripts/;
    mkdir -p generated/;
    echo 'Rewriting'
    python -m plpy.rewrite.expr_lifter $1.py generated/$1_lifted.py;
    echo 'Running and Tracing';
    ipython3 -m plpy.analyze.dynamic_tracer -- generated/$1_lifted.py generated/$1_tracer.pkl --loop_bound 2 --log $1.log;
    echo 'Constructing graph';
    python -m plpy.analyze.graph_builder generated/$1_tracer.pkl generated/$1_graph.pkl --ignore_unknown --memory_refinement 1;
    echo 'Identify donations';
    python -m transfer.identify_donations generated/$1_graph.pkl generated/$1_donations.pkl;
    echo 'Lift donations';
    python -m transfer.lift_donations generated/$1_donations.pkl $1.py generated/$1_functions.pkl
)
