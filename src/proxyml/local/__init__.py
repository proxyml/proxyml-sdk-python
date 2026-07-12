try:
    import proxyml_core.modeling  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "Local challenger training requires the 'local' extra: pip install 'proxyml[local]'"
    ) from exc

from proxyml.local.challenger import LADDERS, Complexity, Rung, TrainedChallenger, train_challenger

__all__ = ["train_challenger", "Complexity", "Rung", "TrainedChallenger", "LADDERS"]
