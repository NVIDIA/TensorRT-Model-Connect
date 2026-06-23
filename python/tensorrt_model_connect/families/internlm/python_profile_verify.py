import einops
import transformers
from transformers.cache_utils import DynamicCache

assert hasattr(DynamicCache, "from_legacy_cache")
print(f"einops={einops.__version__} transformers={transformers.__version__}")
