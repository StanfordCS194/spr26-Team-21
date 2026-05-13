"""Environment loading, optional Anthropic client, and SDV availability flags."""
import os

from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic as _anthropic_mod
    llm = _anthropic_mod.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
except ImportError:
    llm = None

try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    GaussianCopulaSynthesizer = None  # type: ignore[assignment,misc]
    SingleTableMetadata = None  # type: ignore[assignment,misc]
    SDV_AVAILABLE = False
