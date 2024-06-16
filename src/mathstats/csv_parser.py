import csv
from .errors import MathStatsError

def parse_csv_matrix(content: str) -> list:
    try:
        matrix = []
        reader = csv.reader(content.strip().splitlines())
        for r_idx, row in enumerate(reader):
            if not row:
                continue
            float_row = []
            for c_idx, val in enumerate(row):
                val_strip = val.strip()
                if val_strip.lower() == 'nan':
                    float_row.append(float('nan'))
                elif val_strip.lower() in ('inf', 'infinity', '+inf', '+infinity'):
                    float_row.append(float('inf'))
                elif val_strip.lower() in ('-inf', '-infinity'):
                    float_row.append(float('-inf'))
                else:
                    try:
                        float_row.append(float(val_strip))
                    except ValueError:
                        raise MathStatsError(f"Invalid CSV float at row {r_idx}, col {c_idx}: {val}")
            matrix.append(float_row)
        if not matrix:
            raise MathStatsError("Empty input matrix")
        return matrix
    except Exception as e:
        if isinstance(e, MathStatsError):
            raise e
        raise MathStatsError(f"Failed to parse input: {e}")
