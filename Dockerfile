FROM mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /ids

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

CMD ["/bin/bash", "run.sh"]
