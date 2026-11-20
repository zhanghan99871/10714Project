"""Segment policies for the checkpoint / precision policy optimizer."""
from .profiler import SegmentProfile, profile_segments, measure_step, ProfileCache
from .error import SegmentErrors, estimate_segment_errors, measure_precision_error
