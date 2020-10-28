#!/usr/bin/env bash

python3 code.py > fragments.py
yapf -i fragments.py
