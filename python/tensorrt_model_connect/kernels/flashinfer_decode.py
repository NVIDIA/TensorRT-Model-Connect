"""FlashInfer single-decode kernel — JIT compile, register, return .so path."""


def setup(head_dim, dtype=None):
    """Prepare FlashInfer decode kernel for the given head_dim.

    JIT-compiles the kernel for the current GPU, registers it as a TVM-FFI
    global function, and returns the path to the compiled .so file for
    bundle packaging.

    Returns (kernel_name, so_path) where:
        kernel_name: TVM-FFI global function name (e.g. "flashinfer.decode_f16_d64")
        so_path: path to JIT-compiled .so (from FlashInfer cache) for bundle packaging
    """
    import torch
    import tvm_ffi
    import flashinfer.decode as fi_dec

    if dtype is None:
        dtype = torch.float16

    spec = fi_dec.gen_single_decode_module(
        dtype, dtype, dtype, head_dim, head_dim,
        pos_encoding_mode=0,
        use_sliding_window=False,
        use_logits_soft_cap=False,
    )
    mod = spec.build_and_load()

    name = f"flashinfer.decode_f16_d{head_dim}"
    tvm_ffi.register_global_func(name, mod.run, override=True)

    # FlashInfer JIT-compiles to a .so in its cache directory.
    # Return that path directly — no need to export separately.
    so_path = spec.get_library_path()

    return name, so_path
