"""vLLM plugin: prune video patches BEFORE the vision encoder (E9, PLAN.md 15).

Registers a Qwen2.5-VL subclass under the stock architecture name, so the same
checkpoint and arm files load it. With --video-pruning-rate 0 it is the stock model.
"""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration",
                                 "vlm_prune.model:PrunedQwen2_5_VL")
