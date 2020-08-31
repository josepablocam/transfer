from argparse import ArgumentParser


# some lines need to be filtered out of scripts for them to work
def contains_inline(_line):
    # bad matplotlib setting in jupyter that fails in ipython console
    # get_ipython().magic('matplotlib inline')
    return 'inline' in _line


def remove_lines(_file, preds):
    with open(_file, 'r') as f:
        lines = f.readlines()

    clean_lines = [l for l in lines if all(not p(l) for p in preds)]

    with open(_file, 'w') as f:
        for l in clean_lines:
            f.write(l)


def main(args):
    preds = [contains_inline]
    remove_lines(args.input_file, preds)


if __name__ == "__main__":
    parser = ArgumentParser(description='Remove lines that trigger errors')
    parser.add_argument(
        'input_file', type=str, help='File to remove lines from'
    )
    args = parser.parse_args()
    main(args)
