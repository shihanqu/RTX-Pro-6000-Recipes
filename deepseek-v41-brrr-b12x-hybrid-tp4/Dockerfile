ARG BRRR_IMAGE=local/deepseek-v41-brrr-upstream:0bc4dbd0a154
FROM ${BRRR_IMAGE}
COPY patches/hybrid-dispatch.patch /tmp/hybrid-dispatch.patch
COPY tools/patch_decoder.py /tmp/patch_decoder.py
COPY pins.json /tmp/recipe-pins.json
RUN /opt/venv/bin/python /tmp/patch_decoder.py \
    /opt/venv/lib/python3.12/site-packages/brrr /tmp/hybrid-dispatch.patch /tmp/recipe-pins.json
LABEL org.opencontainers.image.title="DeepSeek BRRR/B12X hybrid TP4" \
      org.opencontainers.image.source="https://github.com/shihanqu/RTX-Pro-6000-Recipes" \
      recipe.brrr_revision="0bc4dbd0a15458e0957b899d72c7a5089e3335a5"
