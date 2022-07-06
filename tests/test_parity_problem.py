import datamodules


def test_parity_normal():
    x_dim = 96
    dm = datamodules.ParityDatamodule(
        num_problems=(10000, 1000, 1000),
        vector_size=x_dim,
        extrapolate=False,
        batch_size=32,
    )
    dm.prepare_data()
    dm.setup()

    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        for x, y in dl:
            assert x.size(-1) == x_dim
            assert (y == ((x != 0).sum(dim=-1) % 2)).all()


def test_parity_extrapolate():
    x_dim = 96
    dm = datamodules.ParityDatamodule(
        num_problems=(10000, 1000, 1000),
        vector_size=x_dim,
        extrapolate=True,
        batch_size=32,
    )
    dm.prepare_data()
    dm.setup()

    for dl in [dm.train_dataloader(), dm.val_dataloader()]:
        for x, y in dl:
            assert x.size(-1) == x_dim
            assert ((x != 0).sum(dim=-1) <= (x_dim // 2)).all()
            assert (y == ((x != 0).sum(dim=-1) % 2)).all()

    for x, y in dm.test_dataloader():
        assert x.size(-1) == x_dim
        assert ((x != 0).sum(dim=-1) > x_dim // 2).all()
        assert (y == ((x != 0).sum(dim=-1) % 2)).all()
