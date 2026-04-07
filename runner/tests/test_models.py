from llm_lb.models import LLMParams, MetricSpec, Sample, TaskSpec


def test_task_spec_minimal():
    t = TaskSpec(
        name="t",
        version="1.0",
        metric=MetricSpec(primary="accuracy"),
        labels=["a", "b"],
        llm_params=LLMParams(),
        prompt_template="Q: {prompt}\nA:",
    )
    assert t.metric.primary == "accuracy"
    assert t.llm_params.temperature == 0.0


def test_sample_validation():
    s = Sample(id="x", prompt="hi", expected="a")
    assert s.id == "x"
    assert s.meta == {}
