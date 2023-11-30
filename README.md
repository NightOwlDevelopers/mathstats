# mathstats

Float matrix statistics CLI — parse JSON or CSV matrices and emit indented JSON with per-row and per-column mean and sample standard deviation.

## Features

- `--stat mean|stdev|all` and `--axis rows|columns|all`
- JSON and CSV input (`--input`, default stdin)
- Python `statistics` module semantics for sample stdev (exact rational variance path)
- Additional library modules: distributions, correlation, regression, outlier detection
- Stdlib-only Python 3.8+

## Install

```bash
pip install .
```

## CLI

```bash
mathstats --stat all --axis all --input matrix.json --output -
mathstats --stat mean --axis rows --input data.csv --output stats.json
mathstats --help
mathstats --version
```

## Development

```bash
pip install -e .
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).

## Repository

Maintained by [NightOwlDevelopers](https://github.com/NightOwlDevelopers).
