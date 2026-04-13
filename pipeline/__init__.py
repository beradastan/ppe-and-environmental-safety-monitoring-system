"""Pipeline module"""

from pipeline.complete_pipeline import DetectionPipeline
from pipeline.result_formatter import DetectionResultFormatter

__all__ = ["DetectionPipeline", "DetectionResultFormatter"]
