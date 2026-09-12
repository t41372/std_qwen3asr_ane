"""Keep verified activation expressions intact through Core ML optimization."""


def ane_pass_pipeline():
    import coremltools as ct

    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes(["common::fuse_gelu_exact", "common::fuse_gelu_tanh_approximation"])
    return pipeline


def verify_activation_operators(model) -> None:
    """Fail conversion if optimization reintroduces inaccurate native activations."""
    forbidden = {"gelu", "silu"}
    specification = model.get_spec()
    if specification.WhichOneof("Type") != "mlProgram":
        raise RuntimeError("Activation verification requires an ML Program model")

    def visit(block):
        for operation in block.operations:
            if operation.type in forbidden:
                raise RuntimeError(
                    f"Core ML reintroduced native {operation.type}; the ANE numerical workaround was lost"
                )
            for child in operation.blocks:
                visit(child)

    for function in specification.mlProgram.functions.values():
        for block in function.block_specializations.values():
            visit(block)
