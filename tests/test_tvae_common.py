import pytest

from experiments.tvae.tvae_common import DEFAULT_TVAE_EPOCHS, build_tvae_kwargs


def test_build_tvae_kwargs_materializes_dims_for_cpu():
    out = build_tvae_kwargs(
        {
            "embedding_dim": 128,
            "compress_dim": 256,
            "decompress_dim": 512,
            "batch_size": 1024,
            "learning_rate": 1e-3,
            "l2scale": 1e-5,
            "loss_factor": 2.0,
        },
        epochs=7,
        device="cpu",
    )

    assert DEFAULT_TVAE_EPOCHS == 300
    assert out["epochs"] == 7
    assert out["embedding_dim"] == 128
    assert out["compress_dims"] == (256, 256)
    assert out["decompress_dims"] == (512, 512)
    assert out["batch_size"] == 1024
    assert "learning_rate" not in out
    assert out["l2scale"] == 1e-5
    assert out["loss_factor"] == 2.0
    assert out["cuda"] is False
    assert out["verbose"] is False


def test_build_tvae_kwargs_allows_verbose_cuda():
    out = build_tvae_kwargs(
        {
            "embedding_dim": 64,
            "compress_dim": 128,
            "decompress_dim": 128,
            "batch_size": 256,
            "learning_rate": 2e-4,
            "l2scale": 1e-6,
            "loss_factor": 1.5,
        },
        epochs=3,
        device="cuda",
        verbose=True,
    )

    assert out["cuda"] is True
    assert out["verbose"] is True


@pytest.mark.parametrize("device", ["gpu", "CUDA"])
def test_build_tvae_kwargs_rejects_invalid_device(device):
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        build_tvae_kwargs(
            {
                "embedding_dim": 128,
                "compress_dim": 256,
                "decompress_dim": 512,
                "batch_size": 1024,
                "learning_rate": 1e-3,
                "l2scale": 1e-5,
                "loss_factor": 2.0,
            },
            epochs=7,
            device=device,
        )
