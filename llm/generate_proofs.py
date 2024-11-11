
import sys, os
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{cur_dir}/../")  # for log_config

import string
import time

import colorlog
import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from log_config import get_logger, set_global_log_level
from llm.generate import Generate
from llm.generation_utils import StopStringsCriteria


logger = get_logger(__name__)


class GenerateProofs(Generate):

    def __init__(self, model, tokenizer, device, generation_params):
        super().__init__(model, tokenizer, device, generation_params)

    @torch.no_grad()
    def infer_sample_until_tokens(self, input_ids, model_kwargs, end_strings, deduplicate_newlines=True):
        while True:
            # Prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Infer
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=self.model.generation_config.output_attentions,
                output_hidden_states=self.model.generation_config.output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # Logit processors and warpers
            logits_processor = LogitsProcessorList()
            next_token_scores = logits_processor(input_ids, next_token_logits)
            logits_warper = self.model._get_logits_warper(self.gc)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_token_decoded = self.tokenizer.decode(next_tokens[-1])

            # Fix the case where more newlines are generated
            if deduplicate_newlines and next_token_decoded.count('\n') > 1:
                replace_token = next_token_decoded.replace('\n', '') + '\n'
                replace_token_id = self.tokenizer(replace_token, add_special_tokens=False).input_ids
                next_tokens = torch.tensor(replace_token_id).to(self.device)
                if next_tokens.shape[0] > 1:
                    next_tokens = next_tokens[:1]
            assert next_tokens.shape[0] == 1

            # Update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder)

            # Break if one of the end strings is encountered
            found_end_string = False
            for end_string in end_strings:
                if end_string in next_token_decoded:
                    found_end_string = True
                    break
            if found_end_string:
                break

        return input_ids, model_kwargs

    @torch.no_grad()
    def infer_followandbreak(self, prompt, prefix, follow, separator):
        # Process input and apply prefix
        input_ids, model_kwargs = self.prepare_for_generation(prompt, prefix)

        # Stopping Criteria
        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=self.model.generation_config, stopping_criteria=[StopStringsCriteria(tokenizer=self.tokenizer, stop_strings=self.stop_tokens)]
        )

        # Prepare vars for generation
        eos_token_id_tensor = torch.tensor(self.gc.eos_token_id).to(input_ids.device) if self.gc.eos_token_id is not None else None
        prompt_num_tokens = input_ids[0].shape[0]
        prompt_length = len(self.tokenizer.decode(input_ids[0],skip_special_tokens=True,clean_up_tokenization_spaces=True))
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Prepare follow ids
        follow_ids = self.tokenizer(follow, add_special_tokens=False).input_ids
        follow_ids_pos = 0

        # Prepare newline id
        newline_id = self.tokenizer("\n", add_special_tokens=False).input_ids
        assert len(newline_id) == 1
        newline_id = newline_id[0]

        # Add starting separator
        input_ids, model_kwargs = self.apply_prefix(input_ids, model_kwargs, f"{separator} ")

        # Main decoding loop
        logger.debug(f"temp={self.model.generation_config.temperature} top_p={self.model.generation_config.top_p} max_length={self.model.generation_config.max_length} max_new={self.model.generation_config.max_new_tokens} stop_tokens={self.stop_tokens}")
        s_time = time.time()
        while True:
            # Prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Infer
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=self.model.generation_config.output_attentions,
                output_hidden_states=self.model.generation_config.output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # Sample
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            # Collect all candidates
            follow_token = self.tokenizer.convert_ids_to_tokens(follow_ids[follow_ids_pos])
            follow_token_and_newline_ids = self.tokenizer(f"{follow_token}\n", add_special_tokens=False).input_ids
            candidates = [newline_id, follow_ids[follow_ids_pos]]
            if len(follow_token_and_newline_ids) == 1:
                candidates.append(follow_token_and_newline_ids[0])

            # Remove new-line if the separator was just generated
            separator_ids = self.tokenizer(f"{separator} ", add_special_tokens=False).input_ids
            if input_ids[0,-len(separator_ids):].tolist() == separator_ids:
                candidates.remove(newline_id)

            # Top candidate
            candidates_top_idx = probs[0][candidates].argmax().item()
            next_token = torch.tensor([candidates[candidates_top_idx]]).to(self.device)

            # Update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder)

            # Add separator if new line
            if next_token.item() == newline_id:
                input_ids, model_kwargs = self.apply_prefix(input_ids, model_kwargs, f"{separator} ")
            else:
                follow_ids_pos += 1

            # Break if follow_ids_pos out of scope
            if follow_ids_pos >= len(follow_ids):
                break

            # Break if the remaining tokens are only punctuation
            should_break = True
            for n in range(follow_ids_pos, len(follow_ids)):
                follow_id_decoded = self.tokenizer.decode(follow_ids[n], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                if follow_id_decoded not in string.punctuation:
                    should_break = False
                    break
            if should_break:
                break

        prompt_and_output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = prompt_and_output_text[prompt_length:]

        logger.debug(f"T_main={(time.time() - s_time):.2f}s")

        return output_text

    @torch.no_grad()
    def infer_align(self, prompt, prefix, chunks, separator, end_strings):
        assert len(chunks)

        # Process input and apply prefix
        input_ids, model_kwargs = self.prepare_for_generation(prompt, prefix)

        # Stopping Criteria
        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=self.model.generation_config, stopping_criteria=[StopStringsCriteria(tokenizer=self.tokenizer, stop_strings=self.stop_tokens)]
        )

        # Prepare vars for generation
        eos_token_id_tensor = torch.tensor(self.gc.eos_token_id).to(input_ids.device) if self.gc.eos_token_id is not None else None
        prompt_num_tokens = input_ids[0].shape[0]
        prompt_length = len(self.tokenizer.decode(input_ids[0],skip_special_tokens=True,clean_up_tokenization_spaces=True))
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Main decoding loop
        logger.debug(f"temp={self.model.generation_config.temperature} top_p={self.model.generation_config.top_p} max_length={self.model.generation_config.max_length} max_new={self.model.generation_config.max_new_tokens} stop_tokens={self.stop_tokens}")
        s_time = time.time()
        for chunk in chunks:
            # Add bullet point separator, chunk, arrow, and an opening quotation mark
            input_ids, model_kwargs = self.apply_prefix(input_ids, model_kwargs, f"{separator} {chunk} -> \"")

            # Generate tokens until one of end_strings encountered
            input_ids, model_kwargs = self.infer_sample_until_tokens(input_ids, model_kwargs, end_strings=end_strings)

            # Add the new line token if not present
            if "\n" not in self.tokenizer.decode(input_ids[-1][-1]):
                input_ids, model_kwargs = self.apply_prefix(input_ids, model_kwargs, f"\n")

        prompt_and_output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = prompt_and_output_text[prompt_length:]

        logger.debug(f"T_main={(time.time() - s_time):.2f}s")

        return output_text
