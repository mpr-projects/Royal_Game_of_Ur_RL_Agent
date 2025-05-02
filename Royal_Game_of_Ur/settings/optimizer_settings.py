import optax


def load_optimizer(settings):
    epochs = settings['training']['max_epochs']
    assert epochs > 0

    n_warmup_steps = 15000

    lr_schedule = optax.warmup_cosine_decay_schedule(
        # init_value=1e-6,
        init_value=1e-4,
        peak_value=1e-4,
        warmup_steps=n_warmup_steps,
        decay_steps=epochs-n_warmup_steps,
        end_value=1e-6)

    return optax.adamw(learning_rate=lr_schedule, weight_decay=1e-7)
