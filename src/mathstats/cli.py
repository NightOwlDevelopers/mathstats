import sys
import json
from .errors import MathStatsError
from .matrix import parse_input, compute_stats
from .cli_parser import get_parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    try:
        if args.input == "-":
            content = sys.stdin.read()
        else:
            try:
                with open(args.input, "r") as f:
                    content = f.read()
            except Exception as e:
                raise MathStatsError(f"Could not read input file: {e}")

        matrix = parse_input(content)
        results = compute_stats(matrix, args.stat, args.axis)

        output_str = json.dumps(results, indent=2)

        if args.output == "-":
            sys.stdout.write(output_str + "\n")
        else:
            try:
                with open(args.output, "w") as f:
                    f.write(output_str + "\n")
            except Exception as e:
                raise MathStatsError(f"Could not write output file: {e}")

    except MathStatsError as e:
        sys.stderr.write(f"mathstats: error: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"mathstats: error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
