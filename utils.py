import glob

def get_files(folder_path: str, extension: str = '') -> list[str]:
    return list(glob.glob(f'{folder_path}\*{extension}'))

def parse_int(value: str | list[str]) -> int | list[int] | str:
    try:
        return round(float(value))
    except:
        try:
            return [round(float(v)) for v in value]
        except:
            return value

def parse_float(value: str | list[str]) -> float | list[float] | str:
    try:
        return float(value)
    except:
        try:
            return [float(v) for v in value]
        except:
            return value