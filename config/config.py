import torch


def _validate_config(config, flag):
    """Catch view/config mismatches before model construction."""
    v_num = config["v_num"]
    autoencoder = config["Autoencoder"]
    latent_dim = autoencoder["gcnEncoder1"][-1]

    for i in range(1, v_num + 1):
        gcn_key = f"gcnEncoder{i}"
        graph_key = f"graphEncoder{i}"
        activation_key = f"activations{i}"
        for key in (gcn_key, graph_key, activation_key):
            if key not in autoencoder:
                raise ValueError(f"flag {flag} missing Autoencoder['{key}']")
        if autoencoder[gcn_key][-1] != latent_dim:
            raise ValueError(
                f"flag {flag} has inconsistent latent dim in {gcn_key}: "
                f"{autoencoder[gcn_key][-1]} != {latent_dim}"
            )
        if autoencoder[graph_key][0] != latent_dim:
            raise ValueError(
                f"flag {flag} has {graph_key} input dim "
                f"{autoencoder[graph_key][0]} != latent dim {latent_dim}"
            )

    if autoencoder["graphEncoderf"][0] != latent_dim:
        raise ValueError(
            f"flag {flag} has graphEncoderf input dim "
            f"{autoencoder['graphEncoderf'][0]} != latent dim {latent_dim}"
        )


def get_config(flag=0):
    if flag == 0:
        config = dict(
            dataset="handwritten",
            seed=152,
            mask_seed=5,
            v_num=4,
            topk=10,
            missing_rate=0.7,
            n_clusters=10,
            training=dict(
                epoch=200,
                lr=1e-3,
            ),
            Autoencoder=dict(
                gcnEncoder1=[240, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[76, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder3=[216, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder4=[64, 1024, 1024, 1024, 1024 // 8],

                graphEncoder1=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder2=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder3=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder4=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoderf=[1024 // 8, 1024, 1024, 1024, 1024 // 8],

                activations1="relu",
                activations2="relu",
                activations3="relu",
                activations4="relu",
                activationsf="relu",
                batchnorm=True,
            ),
        )
    elif flag == 1:
        config = dict(
            dataset="100leaves",
            seed=396,
            mask_seed=1,
            v_num=3,
            topk=10,
            missing_rate=0.5,
            n_clusters=100,
            training=dict(
                epoch=1000,
                lr=1e-4,
            ),
            Autoencoder=dict(
                gcnEncoder1=[64, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[64, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder3=[64, 1024, 1024, 1024, 1024 // 2],

                graphEncoder1=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoder2=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoder3=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoderf=[1024 // 2, 1024, 1024, 1024, 1024 // 2],

                activations1="relu",
                activations2="relu",
                activations3="relu",
                activationsf="relu",
                batchnorm=True,
            ),
        )
    elif flag == 2:
        config = dict(
            dataset="Scene-15",
            seed=357,
            mask_seed=1,
            v_num=3,
            topk=10,
            missing_rate=0.5,
            n_clusters=15,
            training=dict(
                epoch=200,
                lr=1e-4,
            ),
            Autoencoder=dict(
                gcnEncoder1=[20, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[59, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder3=[40, 1024, 1024, 1024, 1024 // 2],

                graphEncoder1=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoder2=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoder3=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoderf=[1024 // 2, 1024, 1024, 1024, 1024 // 2],

                activations1="relu",
                activations2="relu",
                activations3="relu",
                activationsf="relu",
                batchnorm=True,
            ),
        )
    elif flag == 3:
        config = dict(
            dataset="LandUse-21",
            seed=110,
            mask_seed=1,
            v_num=3,
            topk=8,
            missing_rate=0.3,
            n_clusters=21,
            training=dict(
                epoch=300,
                lr=2e-5,
            ),
            Autoencoder=dict(
                gcnEncoder1=[20, 1024, 1024, 1024, 1024 // 4],
                gcnEncoder2=[59, 1024, 1024, 1024, 1024 // 4],
                gcnEncoder3=[40, 1024, 1024, 1024, 1024 // 4],

                graphEncoder1=[1024 // 4, 1024, 1024, 1024, 1024 // 4],
                graphEncoder2=[1024 // 4, 1024, 1024, 1024, 1024 // 4],
                graphEncoder3=[1024 // 4, 1024, 1024, 1024, 1024 // 4],
                graphEncoderf=[1024 // 4, 1024, 1024, 1024, 1024 // 4],

                activations1="relu",
                activations2="relu",
                activations3="relu",
                activationsf="relu",
                batchnorm=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset flag: {flag}")

    _validate_config(config, flag)
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config
