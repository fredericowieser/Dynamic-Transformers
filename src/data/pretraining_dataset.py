from typing import Any, Dict
from .base_dataset import BaseDatasetHandler

class PretrainingDataset(BaseDatasetHandler):
    """
    Handles raw text datasets for continued pre-training.
    Inherits all processing logic from BaseDatasetHandler.
    """
    def _process_text_column(self, examples: Dict[str, Any]) -> Dict[str, str]:
        """Extracts and cleans text from the specified column without chat formatting."""
        raw_text = examples.get(self.text_column)
        return {"text": str(raw_text).strip() if raw_text is not None else ""}