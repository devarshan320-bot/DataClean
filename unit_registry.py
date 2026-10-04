UNIT_REGISTRY = {
    "Mass": ["mg", "g", "kg", "tonne"],
    "Length": ["mm", "cm", "m", "km"],
    "Volume": ["ml", "l"],
    "Time": ["ms", "sec", "min", "hour", "day"],
}

UNIT_FACTORS = {
    "Mass": {"mg": 0.001, "g": 1.0, "kg": 1000.0, "tonne": 1000000.0},
    "Length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0},
    "Volume": {"ml": 0.001, "l": 1.0},
    "Time": {"ms": 0.001, "sec": 1.0, "min": 60.0, "hour": 3600.0, "day": 86400.0},
}

def normalize_unit(unit):
    unit = unit.strip().lower()
    if unit in ["sec", "secs", "second", "seconds"]: return "sec"
    if unit in ["min", "mins", "minute", "minutes"]: return "min"
    if unit in ["hour", "hours", "hr", "hrs"]: return "hour"
    if unit in ["day", "days"]: return "day"
    if unit in ["tonne", "tonnes", "t"]: return "tonne"
    if unit in ["l", "liter", "liters", "litre", "litres"]: return "l"
    if unit in ["ml", "milliliter", "milliliters"]: return "ml"
    if unit in ["m", "meter", "meters", "metre", "metres"]: return "m"
    if unit in ["kg", "kgs"]: return "kg"
    if unit in ["g", "gs", "gram", "grams"]: return "g"
    if unit in ["mg", "mgs"]: return "mg"
    if unit in ["cm", "cms"]: return "cm"
    if unit in ["mm", "mms"]: return "mm"
    if unit in ["km", "kms"]: return "km"
    if unit in ["ms"]: return "ms"
    return unit

def get_unit_category(unit):
    normalized = normalize_unit(unit)
    for category, units in UNIT_REGISTRY.items():
        if normalized in units:
            return category
    return "Unknown"

def get_normalized_unit(unit):
    normalized = normalize_unit(unit)
    if get_unit_category(normalized) != "Unknown":
        return normalized
    return None

def convert_value(value, source_unit, target_unit):
    source = normalize_unit(source_unit)
    target = normalize_unit(target_unit)
    cat_source = get_unit_category(source)
    cat_target = get_unit_category(target)
    
    if cat_source != cat_target or cat_source == "Unknown":
        raise ValueError(f"Cannot convert between {source_unit} and {target_unit}")
        
    factors = UNIT_FACTORS[cat_source]
    val_in_base = value * factors[source]
    val_in_target = val_in_base / factors[target]
    
    if val_in_target.is_integer():
        return int(val_in_target)
    
    # Format cleanly
    return round(val_in_target, 6)
