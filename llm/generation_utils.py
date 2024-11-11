
import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class StopStringsCriteria(StoppingCriteria):

    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:

        last_id = input_ids[0][-1]
        last_id_decoded = self.tokenizer.decode(last_id)

        is_done = False
        if self.stop_strings is not None:
            for stop_string in self.stop_strings:
                if stop_string in last_id_decoded:
                    is_done = True
                    break

        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)
