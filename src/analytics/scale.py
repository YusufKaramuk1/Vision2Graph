"""Piksel mesafelerini, config'de olcek verilmisse metre/km cinsine cevirir.

DeepGlobe goruntuleri yaklasik 0.5 m/piksel cozunurluktedir; config.yaml >
scale > meters_per_pixel bu degeri tasir. Donusum, her goruntu icin ayri
georeferans bilgisi olmadigi icin yaklasiktir.
"""


def meters_per_pixel(config):
    """config'den gecerli meters_per_pixel degerini cozer; yoksa None doner."""
    scale = (config or {}).get("scale") or {}
    value = scale.get("meters_per_pixel")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def format_length(pixels, mpp):
    """Piksel uzunlugunu okunabilir metne cevirir.

    mpp None ise 'X px'; degilse 1 km altinda 'X m', uzerinde 'X.XX km'.
    """
    if mpp is None:
        return f"{round(pixels, 1)} px"
    meters = pixels * mpp
    if abs(meters) >= 1000:
        return f"{meters / 1000:.2f} km"
    return f"{meters:.0f} m"
