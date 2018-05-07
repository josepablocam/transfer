import os

def build_script_paths(script_path, output_dir=None):
    script_dir = os.path.dirname(script_path)
    if output_dir is None:
        output_dir = script_dir
    basename = '.'.join(os.path.basename(script_path).split('.')[:-1])
    paths = dict(
        script_path = script_path,
        lifted_path = os.path.join(script_dir, basename + '_lifted.py'),
        trace_path  =  os.path.join(output_dir, basename + '_tracer.pkl'),
        graph_path  = os.path.join(output_dir, basename + '_graph.pkl'),
        donations_path = os.path.join(output_dir, basename + '_donations.pkl'),
        functions_path = os.path.join(output_dir, basename + '_functions.pkl'),
    )
    return paths
