#!/usr/bin/env bash

python3 code.py > fragments.py
# format python code
yapf -i fragments.py
# convert python code to syntax-highlighted html
python pygmentize.py --input fragments.py > fragments.html
