from breakout_rl.train.curriculum import curriculum_params


def test_stage1_easy_then_stage2_real():
    # before switch step -> easy; after -> real
    pw0, bs0 = curriculum_params(step=0, switch_step=1000)
    pw1, bs1 = curriculum_params(step=2000, switch_step=1000)
    assert (pw0, bs0) == (120.0, 200.0)
    assert (pw1, bs1) == (80.0, 300.0)
