"""GPU training, evaluation and inference pipelines for the interview AI.

Sub-packages:
    common      shared infrastructure (config, device, trainer, checkpoint, metrics)
    schemas     dataset-independent data contracts
    datasets    adapters converting raw datasets into ``InterviewSample``
    speech      speech behavior model (handcrafted features / frozen encoder / fine-tuning)
    video       landmark-based temporal video behavior model
    answer      multi-head interview answer evaluator
    multimodal  fusion model skeleton
"""

__version__ = "0.1.0"
