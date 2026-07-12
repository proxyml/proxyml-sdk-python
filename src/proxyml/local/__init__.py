try:
    import proxyml_core.modeling  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "Local challenger training requires the 'local' extra: pip install 'proxyml[local]'"
    ) from exc

from proxyml.local.challenger import (
    LADDERS,
    Complexity,
    Rung,
    TrainedChallenger,
    train_auto_challenger,
    train_challenger,
)

__all__ = [
    "train_challenger",
    "train_auto_challenger",
    "Complexity",
    "Rung",
    "TrainedChallenger",
    "LADDERS",
]
