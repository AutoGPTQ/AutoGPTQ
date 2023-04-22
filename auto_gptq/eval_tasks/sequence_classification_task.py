import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from torch import LongTensor
from transformers import PreTrainedTokenizer, GenerationConfig

from ._base import BaseTask


def levenshtein_distance(str1: str, str2: str):
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(pred: str, classes: List[str]) -> int:
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(pred, class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return min_id


def get_predictions(
    input_ids: LongTensor,
    output_ids: LongTensor,
    num_return_sequences: int,
    tokenizer: PreTrainedTokenizer,
    classes: List[str]
) -> List[int]:
    predictions = []
    for idx, start in enumerate(range(0, len(output_ids), num_return_sequences)):
        sub_output_ids = output_ids[start: start + num_return_sequences]
        sub_generated_ids = sub_output_ids[..., input_ids[idx].size(0):]
        sub_generated_texts = [
            each.lower().strip() for each in tokenizer.batch_decode(
                sub_generated_ids,
                clean_up_tokenization_spaces=True
            )
        ]
        sub_predictions = []
        for gen_text in sub_generated_texts:
            sub_predictions.append(get_closest_label(gen_text, classes))
        predictions.append(Counter(sub_predictions).most_common(1)[0][0])
    return predictions


class SequenceClassificationTask(BaseTask):
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer,
        classes: List[str],
        data_name_or_path: str,
        prompt_col_name: str,
        label_col_name: str,
        device: Optional[str] = None,
        **kwargs
    ):
        kwargs["merge_prompt_label"] = False
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_name_or_path=data_name_or_path,
            prompt_col_name=prompt_col_name,
            label_col_name=label_col_name,
            device=device,
            **kwargs
        )
        self.classes = [each.lower().strip() for each in classes]
        classes_ids = self.tokenizer(classes)
        self.max_new_tokens = max([len(each) for each in classes_ids])

    def _predict(self, batch_data: Dict[str, Any], *args, **kwargs) -> List[int]:
        generation_config = kwargs["generation_config"]
        for key in batch_data:
            if key not in ["input_ids", "attention_mask"]:
                batch_data.pop(key)
        output_ids = self.model.generate(generation_config=generation_config, **batch_data)
        return get_predictions(
            batch_data["input_ids"],
            output_ids,
            generation_config.num_return_sequences,
            self.tokenizer,
            self.classes
        )

    def _parse_labels(self, label_ids: LongTensor) -> List[int]:
        labels = []
        for one_label_ids in label_ids:
            one_label_ids = one_label_ids[(one_label_ids == -100).sum():]
            label = self.classes.index(self.tokenizer.decode(one_label_ids).lower().strip())
            labels.append(label)

        return labels

    def _metric(self, pred: List[int], label: List[int]) -> Dict[str, float]:
        pred = np.array(pred)
        label = np.array(label)

        acc = (pred == label).mean()

        return {"acc": acc}

    def run(self, generation_config: Optional[GenerationConfig] = None) -> Dict[str, float]:
        if not generation_config:
            generation_config = GenerationConfig(
                num_beams=1,
                do_sample=False,
                num_return_sequences=1
            )
        generation_config.max_new_tokens = self.max_new_tokens
        return super().run(generation_config=generation_config)
