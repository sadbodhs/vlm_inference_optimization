"""Measurement harness for VLM inference experiments.

Design rules, inherited from the bar set at sadbodhs.github.io:
  R1  one variable at a time   -> arms are declarative config, never code branches
  R2  saturate before you measure -> every run self-reports whether it was saturated,
                                     and whether the *client* was the bottleneck
  R3  publish the cost, not the win -> accuracy/VRAM travel with every latency number
  R4  say what was not measured -> meta.json records nulls explicitly, not silently
"""
__version__ = "0.1.0"
