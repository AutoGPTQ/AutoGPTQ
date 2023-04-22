from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch

from ._utils import get_dataloader


class BaseTask:
    def __init__(
        self,
        model,
        tokenizer,
        data_name_or_path: str,
        prompt_col_name: str,
        label_col_name: str,
        device: Optional[str] = None,
        **kwargs
    ):
        self.dl = get_dataloader(
            data_name_or_path,
            prompt_col_name=prompt_col_name,
            label_col_name=label_col_name,
            tokenizer=tokenizer,
            **kwargs
        )
        self.model = model
        self.tokenizer = tokenizer

        self.device = device
        if not self.device:
            self.device = self.model.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    @abstractmethod
    def _predict(self, batch_data: Dict[str, Any], **kwargs) -> List[Any]:
        pass

    @abstractmethod
    def _parse_labels(self, label_ids: torch.LongTensor) -> List[Any]:
        pass

    @abstractmethod
    def _metric(self, pred: List[Any], label: List[Any]) -> Dict[str, float]:
        pass

    def run(self, **predict_kwargs) -> Dict[str, float]:
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            predictions = []
            labels = []
            for batch_data in self.dl:
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(self.device)
                predictions += self._predict(batch_data, **predict_kwargs)
                labels += self._parse_labels(batch_data["label"])

        return self._metric(predictions, labels)
