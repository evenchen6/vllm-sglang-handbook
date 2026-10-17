#!/usr/bin/env python3
"""Optional Qwen3.5-4B FP16 forward trace; stop the serving container first.

The 9B compressed serving checkpoint is deliberately not loaded through this HF
example. This smaller, same-generation model is only for five-step cache tracing.
"""
import argparse
import json


def cache_snapshot(cache):
    snapshot = []
    for index, layer in enumerate(getattr(cache, "layers", [])):
        shapes = {}
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            if hasattr(value, "shape"):
                shapes[name] = list(value.shape)
        snapshot.append({"layer": index, "type": type(layer).__name__, "shapes": shapes})
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("This optional FP16 experiment needs a CUDA GPU; see docs/t4-qwen35-lab.md")
    name = "Qwen/Qwen3.5-4B"
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    print(json.dumps({"model": name, "revision": revision, "torch": torch.__version__,
                      "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
                      "use_cache": not args.no_cache}, ensure_ascii=False))
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        name, revision=revision, dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": "用一句话解释 KV cache。"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True).to("cuda")
    all_ids, mask = inputs["input_ids"], inputs["attention_mask"]
    cache = None
    with torch.inference_mode():
        for step in range(args.steps):
            step_ids = all_ids if args.no_cache or step == 0 else all_ids[:, -1:]
            out = model(input_ids=step_ids, attention_mask=mask, past_key_values=cache,
                        use_cache=not args.no_cache)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache = out.past_key_values if not args.no_cache else None
            print(json.dumps({"forward": step, "phase": "prefill" if step == 0 else "decode",
                              "input_shape": list(step_ids.shape), "logits_shape": list(out.logits.shape),
                              "new_token_id": next_id.item(),
                              "new_token_text": tok.decode(next_id[0]),
                              "cache": cache_snapshot(cache)}, ensure_ascii=False))
            # The new token's own cache is computed on the NEXT forward pass.
            all_ids = torch.cat((all_ids, next_id), dim=1)
            mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)
            if next_id.item() in ([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else (tok.eos_token_id or [])):
                break
    print("Cache layouts are implementation-specific; an empty shape list is not zero memory.")


if __name__ == "__main__":
    main()
