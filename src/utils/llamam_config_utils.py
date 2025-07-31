# src/utils/config_utils.py
import logging

log = logging.getLogger(__name__)


def fix_rope_scaling(config):
    """
    Applies defensive fixes to the rope_scaling configuration in a Transformers config object.

    This function ensures rope_scaling is a valid dictionary with required keys ('type', 'rope_type', 'factor').
    It handles common edge cases like missing/invalid configurations, inconsistent keys, or non-dict values.

    Args:
        config: The Transformers config object (e.g., from AutoConfig.from_pretrained).

    Returns:
        The modified config object with fixed rope_scaling.
    """
    if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
        log.warning(
            f"rope_scaling is missing or None. Initializing with default 'linear' type and factor 1.0."
        )
        config.rope_scaling = {"type": "linear", "factor": 1.0, "rope_type": "linear"}
        return config

    if not isinstance(config.rope_scaling, dict):
        log.warning(
            f"rope_scaling is not a dict (was: {type(config.rope_scaling).__name__}). "
            "Overriding with default 'linear' type and factor 1.0."
        )
        config.rope_scaling = {"type": "linear", "factor": 1.0, "rope_type": "linear"}
        return config

    rs = config.rope_scaling

    # Ensure 'type' is set (primary key used in fallbacks)
    if "type" not in rs or rs["type"] is None:
        # Prefer 'rope_type' if available, otherwise default to 'linear'
        new_type = rs.get("rope_type", "linear")
        log.warning(
            f"rope_scaling['type'] is missing or None. Setting to '{new_type}'."
        )
        rs["type"] = new_type

    # Ensure 'rope_type' is set and consistent
    if "rope_type" not in rs or rs["rope_type"] is None:
        log.warning(
            f"rope_scaling['rope_type'] is missing or None. Setting to match 'type': '{rs['type']}'."
        )
        rs["rope_type"] = rs["type"]

    # Ensure 'factor' is set
    if "factor" not in rs or rs["factor"] is None:
        log.warning(f"rope_scaling['factor'] is missing or None. Setting to 1.0.")
        rs["factor"] = 1.0

    # Additional safety: If 'type' and 'rope_type' differ, log and prioritize 'type'
    if rs.get("type") != rs.get("rope_type"):
        log.warning(
            f"rope_scaling 'type' ({rs['type']}) and 'rope_type' ({rs['rope_type']}) differ. "
            "Prioritizing 'type' and setting 'rope_type' to match."
        )
        rs["rope_type"] = rs["type"]

    config.rope_scaling = rs
    log.info(f"Fixed rope_scaling: {rs}")
    return config


def fix_pad_token_id(config):
    """
    Applies defensive fixes to the pad_token_id in a Transformers config object.

    This function handles cases where pad_token_id is a list/tuple (converting to the first int value),
    empty (setting to None), or invalid. It ensures pad_token_id is a single int or None.

    Args:
        config: The Transformers config object (e.g., from AutoConfig.from_pretrained).

    Returns:
        The modified config object with fixed pad_token_id.
    """
    pad_val = getattr(config, "pad_token_id", None)

    if isinstance(pad_val, (list, tuple)):
        if len(pad_val) > 0:
            patched = int(pad_val[0])
            log.warning(
                f"pad_token_id was a list/tuple ({pad_val}). Setting to first value: {patched}."
            )
        else:
            patched = None
            log.warning(f"pad_token_id was an empty list/tuple. Setting to None.")
        config.pad_token_id = patched
    elif pad_val is not None and not isinstance(pad_val, int):
        log.warning(
            f"pad_token_id was invalid type ({type(pad_val).__name__}). Setting to None."
        )
        config.pad_token_id = None

    log.info(f"Fixed pad_token_id: {config.pad_token_id}")
    return config
