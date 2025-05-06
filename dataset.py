from pathlib import Path
import json
import re
import sys
from bs4 import BeautifulSoup
from ftfy import fix_text  

def clean(txt: str) -> str:
    return re.sub(r'\s+', ' ', txt.strip())

def is_number_line(line: str) -> bool:
    return re.fullmatch(r'\d+\.?', line) is not None

def split_into_sections(text: str):
    """find the first line of each number section inside a <pre> tag."""
    first_line = None
    for raw in text.splitlines():
        line = clean(raw)
        if not line:
            continue
        if is_number_line(line):
            if first_line is not None:
                yield first_line
            first_line = None
            continue
        if first_line is None:
            first_line = line
    if first_line is not None:
        yield first_line

def iter_poems(soup: BeautifulSoup):
    """get the pre tags for each poem"""
    for chap in soup.find_all('div', class_='chapter'):
        pres = [pre.get_text() for pre in chap.find_all('pre')]
        if pres:
            yield pres

def fix_encoding(text: str) -> str:
    try:
        text=text.encode('cp1252').decode('utf-8')
        text=fix_text(text)
        return text
    except UnicodeError:
        return fix_text(text)

def main(html_path: Path, out_path: str):
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')

    all_first_lines = []

    for pres in iter_poems(soup):
        for pre_text in pres:
            for first in split_into_sections(pre_text):
                first = fix_encoding(first)
                all_first_lines.append(first)

    if out_path == '-':
        print(json.dumps(all_first_lines, ensure_ascii=False, indent=2))
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_first_lines, f, ensure_ascii=False, indent=2)
        print(f'Wrote {len(all_first_lines)} lines → {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3 or not Path(sys.argv[1]).is_file():
        sys.exit('Usage: python whitman_first_lines_to_json.py whitman.html output.json|-\n'
                 '       (use "-" for stdout)')
    main(Path(sys.argv[1]), sys.argv[2])
