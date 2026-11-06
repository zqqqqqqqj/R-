"""Domain-independent memory configuration."""
from dataclasses import asdict, dataclass
import math
import struct

WRITER_VERSION = 'bounded_count_convex_v2'
MEMORY_INTERFACE_VERSION = 'separate_capped_v2'

def validate_writer_bound(value, storage_dtype='float32'):
    """Require an exact finite endpoint, without importing a model or torch.

    If B is a representable endpoint, rounding an in-range FP32 merge to the
    configured persistent dtype cannot round past B. In particular, 0.1 must
    not silently become a larger BF16 bound than the configured number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('writer_bound must be a positive finite number')
    if storage_dtype not in {'float32', 'bfloat16'}:
        raise ValueError('Unsupported writer bound storage dtype')
    try:
        packed = struct.pack('>f', value)
    except (OverflowError, struct.error) as error:
        raise ValueError('writer_bound must be exactly representable in its storage dtype') from error
    exact_fp32 = struct.unpack('>f', packed)[0]
    bits = int.from_bytes(packed, 'big')
    if exact_fp32 != value or (storage_dtype == 'bfloat16' and bits & 0xFFFF):
        raise ValueError('writer_bound must be exactly representable in its storage dtype')
    return float(value)


@dataclass(frozen=True)
class MemoryConfig:
    prefix_dim: int
    state_dim: int
    camera_indices: tuple[int, ...]
    coarse_dim: int
    slots: int = 8
    residual_dim: int = 128
    query_dim: int = 256
    read_dim: int = 128
    hidden_dim: int = 256
    image_grid: int = 16
    storage_dtype: str = 'bfloat16'
    merge_rule: str = 'ward'
    time_scale: float = 2500.0
    teacher_seed: int = 20261016
    teacher_scale_floor: float = .01
    writer_bound: float = 1.0
    relative_energy_floor: float = 1e-12
    readout_rms_cap: float = 4.0
    memory_token_rms_cap: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'camera_indices', tuple(self.camera_indices))
        for key in ('prefix_dim', 'state_dim', 'coarse_dim', 'slots', 'residual_dim', 'query_dim',
                    'read_dim', 'hidden_dim', 'image_grid'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        cameras = self.camera_indices
        if (not cameras or any(type(value) is not int or value < 0 for value in cameras)
                or len(cameras) != len(set(cameras))):
            raise ValueError('camera_indices must contain distinct nonnegative camera slots')
        if self.image_grid % 2 or self.coarse_dim % (4*len(cameras)):
            raise ValueError('Use an even image grid and coarse_dim divisible by 4*camera_count')
        if self.coarse_dim//(4*len(cameras)) > self.prefix_dim:
            raise ValueError('The per-region projection width cannot exceed prefix_dim')
        if self.storage_dtype not in {'bfloat16', 'float32'} or self.merge_rule not in {'ward','shortest','cosine'}:
            raise ValueError('Unsupported storage dtype or merge rule')
        for key in ('time_scale', 'teacher_scale_floor', 'relative_energy_floor', 'readout_rms_cap',
                    'memory_token_rms_cap'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if type(self.teacher_seed) is not int or self.teacher_seed < 0:
            raise ValueError('teacher_seed must be a nonnegative integer')
        validate_writer_bound(self.writer_bound, self.storage_dtype)

    def to_dict(self):
        return asdict(self)
