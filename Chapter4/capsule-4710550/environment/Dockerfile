# hash:sha256:58a1c1cfab1b739af908e361a9df6ba70f545167cf3fcf03db3c5d247ccd0eed
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==4.2.3 \
        libiconv==1.16 \
        rpy2==3.5.11 \
        tzlocal==5.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    jupyter==1.0.0 \
    matplotlib==3.9.1 \
    numpy==2.0.0 \
    pandas==2.2.2 \
    psutil==6.0.0 \
    pydoe2==1.3.0 \
    pyod==2.0.1 \
    scikit-learn==1.5.1 \
    seaborn==0.13.2 \
    sklearn==0.0

COPY postInstall /
RUN /postInstall
