from .json_parser import parse_json_matrix
from .csv_parser import parse_csv_matrix
from .stats import compute_stats
from .utils import is_matrix_empty

def parse_input(content: str) -> list:
    res = parse_json_matrix(content)
    if res is not None:
        return res
    return parse_csv_matrix(content)
