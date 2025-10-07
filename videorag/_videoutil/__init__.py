from .split import split_video, saving_video_segments
from .asr import speech_to_text
from .caption import segment_caption, merge_segment_information, retrieved_segment_caption

# Lazy wrappers to avoid importing heavy dependencies unless needed
def encode_video_segments(*args, **kwargs):
    from .feature import encode_video_segments as _impl
    return _impl(*args, **kwargs)

def encode_string_query(*args, **kwargs):
    from .feature import encode_string_query as _impl
    return _impl(*args, **kwargs)