
import sys, os
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{cur_dir}/../")  # for log_config

import time

import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from log_config import get_logger, set_global_log_level
from llm.generation_utils import StopStringsCriteria


logger = get_logger(__name__)


class Generate:

    def __init__(self, model, tokenizer, device, generation_params):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_params = generation_params

        self.set_params(**generation_params)
        self.stop_tokens = generation_params.get('stop_tokens', None)

        self.prefix_method = "merge"
        # self.prefix_method = "apply"

    def set_params(self, **kwargs):
        self.gc = GenerationConfig(
            bos_token_id=128000,
            do_sample=True,
            eos_token_id=[128001, 128009],
            max_length=kwargs.get('max_length', 8192),
            max_new_tokens=kwargs.get('max_new_tokens', None),
            pad_token_id=128001,
            temperature=max(kwargs.get('temperature', 0.6), 0.0001),
            top_p=kwargs.get('top_p', 0.9)
        )

    @torch.no_grad()
    def apply_prefix(self, input_ids, model_kwargs, prefix):
        prefix_tokens = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        assert prefix_tokens.shape[0] == 1
        for n in range(prefix_tokens.shape[1]):
            expected_token_id = prefix_tokens[0][n]

            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=self.model.generation_config.output_attentions,
                output_hidden_states=self.model.generation_config.output_hidden_states,
            )
            next_tokens = expected_token_id.reshape([1])
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder)

        return input_ids, model_kwargs

    @torch.no_grad()
    def prepare_for_generation(self, prompt, prefix=None):
        s_time = time.time()

        # Update generation config
        self.model.generation_config = self.gc

        # Prefix - merge
        if prefix and self.prefix_method == "merge":
            prompt += prefix

        # Prepare input tokens
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]

        # Model kwargs
        model_kwargs = {'attention_mask': attn_masks, 'use_cache': True}

        # Prefix
        if prefix and self.prefix_method == "apply":
            input_ids, model_kwargs = self.apply_prefix(input_ids, model_kwargs, prefix)

        # Prepare max_length
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = self.generation_params.get("max_length") is None and self.model.generation_config.max_length is not None
        if self.model.generation_config.max_new_tokens is not None:
            self.model.generation_config.max_length = self.model.generation_config.max_new_tokens + input_ids_length

        logger.debug(f"T_preprocess={(time.time() - s_time):.2f}s")

        return input_ids, model_kwargs

    @torch.no_grad()
    def infer_sample(self, prompt, prefix=None):
        # Process input and apply prefix
        input_ids, model_kwargs = self.prepare_for_generation(prompt, prefix)

        # Stopping Criteria
        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=self.model.generation_config, stopping_criteria=[StopStringsCriteria(tokenizer=self.tokenizer, stop_strings=self.stop_tokens)]
        )

        # Prepare vars for generation
        eos_token_id_tensor = torch.tensor(self.gc.eos_token_id).to(input_ids.device) if self.gc.eos_token_id is not None else None
        scores = None
        this_peer_finished = False
        prompt_num_tokens = input_ids[0].shape[0]
        prompt_length = len(self.tokenizer.decode(input_ids[0],skip_special_tokens=True,clean_up_tokenization_spaces=True))
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

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

            # Logit processors and warpers
            logits_processor = LogitsProcessorList()
            next_token_scores = logits_processor(input_ids, next_token_logits)
            logits_warper = self.model._get_logits_warper(self.gc)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Finished sentences should have their next token be a padding token
            if self.gc.eos_token_id is not None:
                if self.gc.pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + self.gc.pad_token_id * (1 - unfinished_sequences)

            # Update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder)

            # If eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # Stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # Stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished:
                break

        prompt_and_output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = prompt_and_output_text[prompt_length:]

        logger.debug(f"T_main={(time.time() - s_time):.2f}s")
        logger.debug(f"T_token={(time.time() - s_time)/(input_ids[0].shape[0]-prompt_num_tokens):.2f}s")

        return output_text

    @torch.no_grad()
    def infer_hypotheses(self, prompt, prefix, hypotheses):
        # Process input and apply prefix
        input_ids, model_kwargs = self.prepare_for_generation(prompt, prefix)
        assert input_ids.shape[0] == 1

        # Prepare vars for generation
        eos_token_id_tensor = torch.tensor(self.gc.eos_token_id).to(input_ids.device) if self.gc.eos_token_id is not None else None
        prompt_num_tokens = input_ids[0].shape[0]
        prompt_length = len(self.tokenizer.decode(input_ids[0],skip_special_tokens=True,clean_up_tokenization_spaces=True))
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Repeat
        hyp_input_ids = input_ids.repeat(len(hypotheses), 1)

        # Tokenise hypotheses
        tokenised_hypotheses = self.tokenizer(hypotheses, add_special_tokens=False).input_ids
        tokenised_hypotheses_maxlen = max(list(map(len, tokenised_hypotheses)))

        # Main decoding loop
        logger.debug(f"temp={self.model.generation_config.temperature} top_p={self.model.generation_config.top_p} max_length={self.model.generation_config.max_length} max_new={self.model.generation_config.max_new_tokens} stop_tokens={self.stop_tokens}")
        s_time = time.time()
        results = [(n, {'tokens': []}) for n in hypotheses]
        for dec_step in range(tokenised_hypotheses_maxlen):
            # Prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(hyp_input_ids, **model_kwargs)

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

            # Collect probs for hypotheses tokens
            next_token_ids = []
            for hyp_n in range(len(tokenised_hypotheses)):
                if dec_step < len(tokenised_hypotheses[hyp_n]):
                    hyp_token = tokenised_hypotheses[hyp_n][dec_step]
                    probs_hyp_token = probs[hyp_n][hyp_token]

                    results[hyp_n][1]['tokens'].append((
                        self.tokenizer.decode(hyp_token),
                        probs_hyp_token.item(),
                        probs_hyp_token.log().item()
                    ))
                else:
                    hyp_token = self.gc.pad_token_id
                next_token_ids.append(hyp_token)
            next_token_ids = torch.tensor(next_token_ids).to(self.device)

            # Update generated ids, model inputs, and length for next step
            hyp_input_ids = torch.cat([hyp_input_ids, next_token_ids.reshape(-1, 1)], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder)

        # Compute log total and mean
        for r in results:
            r[1]['log_total'] = sum([n[2] for n in r[1]['tokens']])
            r[1]['log_mean'] = r[1]['log_total'] / len(r[1]['tokens'])

        # Print total time
        logger.debug(f"T_main={(time.time() - s_time):.2f}s")

        return results

    @torch.no_grad()
    def infer_yesno(self, prompt, prefix, return_token_probs):
        results = self.infer_hypotheses(prompt, prefix, ["Yes", "No"])
        results_sorted = list(sorted(results, key=lambda x: x[1]['log_total'], reverse=True))
        best_hypothesis = results_sorted[0][0]

        # Token probabilities
        if return_token_probs:
            token2prob = {}

            # Collect probabilities of first tokens
            for r in results:
                token2prob[r[0]] = r[1]['tokens'][0][1]

            # Normalise
            token2prob_sum = sum(token2prob.values())
            for k in token2prob:
                token2prob[k] /= token2prob_sum

            return best_hypothesis, token2prob
        else:
            return best_hypothesis
