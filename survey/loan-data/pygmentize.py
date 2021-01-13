from argparse import ArgumentParser
import math
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

import pdb
#


def to_html(code):
    html_code = highlight(
        code, PythonLexer(), HtmlFormatter(
            noclasses=True,
            style="default",
        )
    )

    html_ext_code = []
    for line in html_code.split("\n"):
        if "Fragment" in line:
            if len(html_ext_code) > 0:
                html_ext_code.append("</pre>")
            html_ext_code.append("<pre>")
        html_ext_code.append(line)
    return "\n".join(html_ext_code)


#     # add <br>
#     html_code = "<br>\n".join(html_code.split("\n"))
#     html_code = html_code.replace("\t", "&#9;")
#     return html_code
#
#
def code_replace_tabs(src):
    new_code = []
    for line in src.split("\n"):
        trimmed = line.lstrip()
        num = len(line) - len(trimmed)
        num_tabs = math.ceil(num / 2)
        tab_line = (num_tabs * "\t") + trimmed
        new_code.append(tab_line)
    return "\n".join(new_code)


def get_args():
    parser = ArgumentParser(description="Pygmentize")
    parser.add_argument("--input", type=str, help="Source code")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.input, "r") as fin:
        src = fin.read()

    # src = code_replace_tabs(src)
    html_code = to_html(src)
    print(html_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
