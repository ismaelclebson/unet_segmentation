# Requisitos
Python 3.11

PyTorch 2.4.0+ (com suporte a CUDA, se disponível)

Então recomendamos [usar mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
para instalar as dependências.

    mamba env create --file environment.yml

> [!NOTE]
> O comando acima foi testado em dispositivos Linux com GPUs CUDA.

Para executar o script de treinamento.

    python -m src.scripts.train
