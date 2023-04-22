import math
from typing import Any, Dict, List

from torch import LongTensor

from ._base import BaseTask


class LanguageModelingTask(BaseTask):
    def predict(self, batch_data: Dict[str, Any], *args, **kwargs) -> List[Any]:
        outputs = self.model(**batch_data)
        loss = outputs.loss.view([1, -1]).squeeze().cpu().numpy().tolist()

        return loss

    def parse_labels(self, label_ids: LongTensor) -> List[Any]:
        return []

    def metric(self, pred: List[Any], label: List[Any]) -> Dict[str, float]:
        return {"ppl": math.exp(sum(pred) / len(pred))}

    def run(self) -> Dict[str, float]:
        return super().run()
