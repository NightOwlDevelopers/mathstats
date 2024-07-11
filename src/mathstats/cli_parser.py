import argparse
from .config import VERSION, DESCRIPTION, STAT_HELP, AXIS_HELP, INPUT_HELP, OUTPUT_HELP

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mathstats", description=DESCRIPTION)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--stat", choices=["mean", "stdev", "all"], default="all", help=STAT_HELP)
    parser.add_argument("--axis", choices=["rows", "columns", "all"], default="all", help=AXIS_HELP)
    parser.add_argument("--input", default="-", help=INPUT_HELP)
    parser.add_argument("--output", default="-", help=OUTPUT_HELP)
    return parser
