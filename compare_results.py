import json
from pathlib import Path
from pprint import pprint

def load_results(path="results.json"):
    """Return the parsed JSON list."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Couldn’t find {path.resolve()}")
    with path.open(encoding='utf-8') as f:
        return json.load(f)

def choose_condition(results):
    print("\nAvailable experimental conditions:\n")
    for i, (param_str, _, _) in enumerate(results):
        print(f"[{i:>2}]  {param_str}")

    while True:
        raw = input(
            "\nPick a condition (index number or copy-paste the param string): "
        ).strip()
        # Allow selection by index
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(results):
                return idx
        else:  # selection by full param string
            for i, (param_str, _, _) in enumerate(results):
                if raw == param_str:
                    return i
        print("Not a valid choice, try again.")

def show_condition(param_str, texts, stats):
    line = "-" * 80
    print(f"\n{line}\nCONDITION:\n{param_str}\n{line}")

    print("\nGenerated responses:")
    for j, txt in enumerate(texts, 1):
        print(f"\n— Response {j} —\n{txt}")

    print(f"\n{line}\nQuantitative statistics\n{line}")
    pprint(stats, sort_dicts=False)
    print("\n")

def main():
    results = load_results()
    idx = choose_condition(results)
    param_str, responses, metrics = results[idx]
    show_condition(param_str, responses, metrics)

if __name__ == "__main__":
    main()
