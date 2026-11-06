ARG BRRR_IMAGE=local/deepseek-v41-brrr-upstream:0bc4dbd0a154
FROM ${BRRR_IMAGE}
COPY patches/ /tmp/recipe-patches/
COPY tools/patch_brrr.py tools/test_ram_backend.py /tmp/
COPY pins.json /tmp/recipe-pins.json
COPY kernels/engram_ram.cpp /opt/venv/lib/python3.12/site-packages/brrr/kernels/engram_ram.cpp
RUN /opt/venv/bin/python /tmp/patch_brrr.py \
    /opt/venv/lib/python3.12/site-packages/brrr /tmp/recipe-patches /tmp/recipe-pins.json
RUN --network=none MAX_JOBS=1 /opt/venv/bin/python /tmp/test_ram_backend.py
LABEL org.opencontainers.image.title="DeepSeek BRRR/B12X hybrid TP4 (RAM Engrams)" \
      org.opencontainers.image.source="https://github.com/shihanqu/RTX-Pro-6000-Recipes" \
      recipe.brrr_revision="0bc4dbd0a15458e0957b899d72c7a5089e3335a5"
