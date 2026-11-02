import pytest

from lsa.training.native.config import (
    AttentionConfig,
    CheckpointConfig,
    DataConfig,
    EvaluationConfig,
    MixtureChild,
    ModelConfig,
    NativeConfig,
    OptimConfig,
    RunConfig,
)


def test_the_defaults_are_the_flat_design_read_at_16k():
    config = AttentionConfig()
    assert (config.window_size, config.sink_size, config.top_k) == (512, 4, 1024)
    assert (config.pool_size, config.bucket_size, config.bucket_cap, config.block_size) == (4, 8, 16, 512)
    assert config.buckets_per_block == 64


@pytest.mark.parametrize(("top_k", "top_p"), [(4096, 1024), (2048, 512), (1024, 256)])
def test_top_p_follows_the_designs_rule_of_twice_the_buckets_top_k_fills(top_k, top_p):
    assert AttentionConfig(top_k=top_k).selection_width == top_p


def test_a_top_p_the_run_names_beats_the_rule():
    assert AttentionConfig(top_k=4096, top_p=7).selection_width == 7


def test_the_key_budget_is_sinks_window_and_top_k():
    assert AttentionConfig().key_budget == 4 + 512 + 1024


@pytest.mark.parametrize("field", ["kind", "backend", "partition", "rank"])
def test_a_name_that_is_not_an_option_is_refused(field):
    with pytest.raises(ValueError, match=field):
        AttentionConfig(**{field: "nonsense"})


def test_a_cap_under_twice_the_bucket_size_is_refused_where_ward_needs_the_room():
    with pytest.raises(ValueError, match="twice"):
        AttentionConfig(bucket_cap=15)
    assert AttentionConfig(bucket_cap=15, partition="chunks").bucket_cap == 15


def test_sizes_that_do_not_nest_are_refused():
    with pytest.raises(ValueError, match="divide block_size"):
        AttentionConfig(bucket_size=7)
    with pytest.raises(ValueError, match="pool_size"):
        AttentionConfig(pool_size=3)


def test_the_dense_and_window_arms_do_not_care_about_the_buckets():
    assert AttentionConfig(kind="window", bucket_size=7, window_size=1540, sink_size=0).window_size == 1540


def test_evaluation_edges_must_increase():
    with pytest.raises(ValueError, match="increasing"):
        EvaluationConfig(edges=[512, 512])


def test_a_trunk_to_branch_from_must_be_an_absolute_path():
    with pytest.raises(ValueError, match="init_from"):
        RunConfig(output="/tmp/native", init_from="trunk/checkpoint/step-3000")
    assert RunConfig(output="/tmp/native", init_from="/trunk/checkpoint/step-3000").init_from.endswith("3000")


def test_a_checkpoint_to_score_must_be_an_absolute_path():
    with pytest.raises(ValueError, match="run.evaluation.checkpoint"):
        EvaluationConfig(checkpoint="released/checkpoint/step-5866")
    assert EvaluationConfig(checkpoint="/released/checkpoint/step-5866").checkpoint.endswith("5866")


def test_an_evaluation_scores_the_checkpoint_the_command_line_names_before_the_one_the_config_names():
    named = NativeConfig(
        run=RunConfig(
            output="/tmp/native", name="scored", evaluation=EvaluationConfig(checkpoint="/a/step-1")
        )
    )
    assert named.scored_checkpoint() == "/a/step-1"
    assert named.scored_checkpoint("/b/step-2") == "/b/step-2"


def test_an_evaluation_that_names_no_checkpoint_anywhere_is_refused_saying_what_to_pass():
    unnamed = NativeConfig(run=RunConfig(output="/tmp/native", name="scored"))
    with pytest.raises(ValueError, match="--checkpoint") as raised:
        unnamed.scored_checkpoint()
    assert "scored" in str(raised.value) and "run.evaluation.checkpoint" in str(raised.value)
    assert unnamed.scored_checkpoint("/b/step-2") == "/b/step-2"


def test_the_retrieval_probe_defaults_to_six_distances_from_inside_the_window_to_the_far_end():
    config = EvaluationConfig()
    assert config.retrieval_distances == [256, 1024, 2048, 4096, 8192, 15000]
    assert (config.retrieval_sequences, config.retrieval_distractors) == (32, 4)


def test_retrieval_distances_must_increase_and_none_switches_the_probe_off():
    with pytest.raises(ValueError, match="retrieval_distances"):
        EvaluationConfig(retrieval_distances=[1024, 256])
    with pytest.raises(ValueError, match="retrieval_distances"):
        EvaluationConfig(retrieval_distances=[0, 256])
    assert EvaluationConfig(retrieval_distances=[]).retrieval_distances == []


def test_retrieval_counts_must_make_sense():
    with pytest.raises(ValueError, match="retrieval_sequences"):
        EvaluationConfig(retrieval_sequences=0)
    with pytest.raises(ValueError, match="retrieval_distractors"):
        EvaluationConfig(retrieval_distractors=-1)


def test_a_held_out_set_must_be_whole_batches_on_every_rank():
    config = EvaluationConfig(every=500, sequences=256)
    config.check_ranks(8, 1)
    with pytest.raises(ValueError, match="multiple of 192"):
        config.check_ranks(24, 8)
    EvaluationConfig(every=0, sequences=3).check_ranks(8, 1)


def test_the_lsa_arm_reads_no_summaries_and_clusters_with_ward():
    with pytest.raises(ValueError, match="lsa"):
        AttentionConfig(kind="lsa")
    with pytest.raises(ValueError, match="lsa"):
        AttentionConfig(kind="lsa", coarse=False, partition="chunks")
    assert AttentionConfig(kind="lsa", coarse=False).selection_width == 256


def test_a_packing_that_is_not_an_option_is_refused():
    with pytest.raises(ValueError, match="packing"):
        DataConfig(packing="shuffle")


def test_a_mixture_weight_must_be_positive():
    with pytest.raises(ValueError, match="weights must be positive"):
        DataConfig(mixture=[MixtureChild("gpt2"), MixtureChild("fineweb_edu_100bt", 0.0)])


def test_a_mixture_replaces_the_name_as_what_the_run_trains_on():
    data = DataConfig(name="gpt2", mixture=[MixtureChild("fineweb_edu_100bt", 3.0), MixtureChild("gpt2")])
    assert data.corpora == [("fineweb_edu_100bt", 3.0), ("gpt2", 1.0)]
    assert DataConfig(name="gpt2").corpora == [("gpt2", 1.0)]


def test_the_held_out_pass_needs_a_tokenized_corpus():
    run = RunConfig(output="/tmp/native", optim=OptimConfig(tokens_per_rank=4096))
    run.evaluation.every = 100
    with pytest.raises(ValueError, match="held-out pass reads a tokenized corpus"):
        NativeConfig(data=DataConfig(name="fineweb_edu_100bt", seq_len=4096), run=run)
    run.evaluation.every = 0
    assert NativeConfig(data=DataConfig(name="fineweb_edu_100bt", seq_len=4096), run=run).data.asset.name


def test_seq_len_must_divide_every_tokenized_row_of_the_mixture():
    run = RunConfig(output="/tmp/native", optim=OptimConfig(tokens_per_rank=4096))
    run.evaluation.every = 0
    with pytest.raises(ValueError, match="seq_len must divide the tokenized row"):
        NativeConfig(
            data=DataConfig(name="fineweb_edu_100bt", mixture=[MixtureChild("gpt2")], seq_len=3000), run=run
        )


def test_a_model_names_no_head_shape_unless_a_run_does():
    model = ModelConfig()
    assert (model.query_head_count, model.kv_head_count, model.head_dim, model.rotary_dim) == (None,) * 4


@pytest.mark.parametrize("field", ["query_head_count", "kv_head_count", "head_dim"])
def test_a_head_count_or_dim_that_is_not_positive_is_refused(field):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: 0})


def test_a_rotation_may_be_none_at_all_but_not_negative():
    assert ModelConfig(rotary_dim=0).rotary_dim == 0
    with pytest.raises(ValueError, match="rotary_dim"):
        ModelConfig(rotary_dim=-1)


def test_query_heads_have_to_split_into_whole_groups_over_the_key_value_heads():
    assert ModelConfig(query_head_count=8, kv_head_count=1).kv_head_count == 1
    with pytest.raises(ValueError, match="kv_head_count"):
        ModelConfig(query_head_count=8, kv_head_count=3)


def run_config(**model):
    return NativeConfig(
        model=ModelConfig(**model),
        data=DataConfig(name="gpt2", seq_len=4096),
        run=RunConfig(output="/tmp/native"),
    )


def test_the_tokenizer_is_the_corpus_s_unless_the_model_names_one():
    assert run_config().tokenizer == "gpt2"
    assert run_config(tokenizer="qwen35_4b").tokenizer == "qwen35_4b"


def test_the_tokenizer_of_a_qwen_corpus_is_qwens():
    config = NativeConfig(data=DataConfig(name="qwen3_5", seq_len=4096), run=RunConfig(output="/tmp/native"))
    assert config.tokenizer == "qwen35_4b"


def test_a_step_has_to_be_whole_passes_of_every_rank():
    kwargs = {"data": DataConfig(name="gpt2", seq_len=4096), "run": RunConfig(output="/tmp/n", dp_shard=2)}
    kwargs["run"].optim = OptimConfig(tokens_per_rank=8192, tokens_per_step=32768)
    assert NativeConfig(**kwargs).run.optim.tokens_per_step == 32768
    kwargs["run"].optim = OptimConfig(tokens_per_rank=8192, tokens_per_step=24576)
    with pytest.raises(ValueError, match="tokens_per_step"):
        NativeConfig(**kwargs)


def test_a_decay_shape_that_is_not_an_option_is_refused():
    with pytest.raises(ValueError, match="decay_shape"):
        OptimConfig(decay_shape="exp")


@pytest.mark.parametrize("decay_ratio", [-0.1, 1.1])
def test_a_decay_ratio_outside_the_run_is_refused(decay_ratio):
    with pytest.raises(ValueError, match="decay_ratio"):
        OptimConfig(decay_ratio=decay_ratio)


def test_a_warmup_longer_than_the_run_is_refused():
    with pytest.raises(ValueError, match="warmup_steps"):
        OptimConfig(steps=100, warmup_steps=101)


def test_the_decay_start_is_where_the_stable_stage_ends():
    optim = OptimConfig(steps=1000, warmup_steps=50, decay_ratio=0.2)
    assert (optim.decay_step_count, optim.decay_start_step) == (200, 800)


def test_a_decay_start_checkpoint_needs_a_decay():
    with pytest.raises(ValueError, match="decay_start"):
        RunConfig(
            output="/tmp/native",
            optim=OptimConfig(decay_ratio=0.0),
            checkpoint=CheckpointConfig(decay_start=True),
        )
    run = RunConfig(
        output="/tmp/native",
        optim=OptimConfig(decay_ratio=0.2),
        checkpoint=CheckpointConfig(decay_start=True),
    )
    assert run.checkpoint.decay_start


@pytest.mark.parametrize("factor", [-0.1, 1.1])
def test_a_learning_rate_floor_outside_the_peak_is_refused(factor):
    with pytest.raises(ValueError, match="min_lr_factor"):
        OptimConfig(min_lr_factor=factor)


def test_a_context_can_span_whole_tokenized_rows():
    config = NativeConfig(
        data=DataConfig(name="gpt2", seq_len=131072),
        run=RunConfig(output="/tmp/native", optim=OptimConfig(tokens_per_rank=131072)),
    )
    assert config.data.seq_len == 131072


@pytest.mark.parametrize("ratio", [0.8, 0.95, 1.0])
def test_a_decay_start_checkpoint_requires_a_stable_stage_after_warmup(ratio):
    with pytest.raises(ValueError, match="stable stage"):
        RunConfig(
            output="/tmp/native",
            optim=OptimConfig(steps=100, warmup_steps=20, decay_ratio=ratio),
            checkpoint=CheckpointConfig(decay_start=True),
        )


def test_cosine_cannot_request_a_stable_stage_checkpoint():
    with pytest.raises(ValueError, match="schedule=wsd"):
        RunConfig(
            output="/tmp/native",
            optim=OptimConfig(schedule="cosine"),
            checkpoint=CheckpointConfig(decay_start=True),
        )


def test_an_unknown_schedule_cannot_be_loaded():
    with pytest.raises(ValueError, match="run.optim.schedule"):
        OptimConfig(schedule="unknown")
