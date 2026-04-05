from tinyserve.expert_execution import ExpertPipeline


def test_run_layer_method_exists():
    assert hasattr(ExpertPipeline, "run_layer"), "ExpertPipeline must have run_layer"


def test_run_layer_batched_method_exists():
    assert hasattr(ExpertPipeline, "run_layer_batched")


def test_prefetch_for_next_token_method_exists():
    assert hasattr(ExpertPipeline, "prefetch_for_next_token")
