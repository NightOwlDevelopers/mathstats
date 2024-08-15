import json
from .errors import MathStatsError

def parse_json_matrix(content: str) -> list:
    if content.strip().startswith('['):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                if not data:
                    raise MathStatsError("Empty input matrix")
                matrix = []
                for r_idx, row in enumerate(data):
                    if not isinstance(row, list):
                        raise MathStatsError(f"Row {r_idx} is not a list")
                    float_row = []
                    for c_idx, val in enumerate(row):
                        try:
                            if val is None:
                                float_row.append(float('nan'))
                            else:
                                float_row.append(float(val))
                        except (ValueError, TypeError):
                            raise MathStatsError(f"Invalid number at row {r_idx}, col {c_idx}: {val}")
                    matrix.append(float_row)
                return matrix
        except json.JSONDecodeError as e:
            raise MathStatsError(f"Failed to parse JSON: {e}")
    return None
