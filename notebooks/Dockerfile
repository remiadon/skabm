# skabm's live simulation pages as a Hugging Face Space (Docker SDK): every notebook
# built on skabm.player, found at build time, so a new page needs no edit here.
# A Bayonne run peaks at 4-7 GB; a free CPU Space has 16 GB.  The Space holds only this
# file and its README: skabm comes from GitHub at build time, so a "Factory rebuild" of
# the Space picks up main.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user

RUN git clone --depth 1 https://github.com/remiadon/skabm.git \
    && pip install --no-cache-dir --user "./skabm[voila]"

WORKDIR /home/user/skabm/notebooks
# The pages only: a notebook that imports skabm.player is one; the others leave the
# image, since Voilà would run any notebook opened from its tree.  Each page is executed
# once, which sets it up without simulating (that waits for Run), so whatever it
# downloads is in the image and a restart doesn't refetch it.  A page that fails to set
# up fails the build.
RUN for notebook in *.ipynb; do \
        if grep -q "skabm.player" "$notebook"; then jupyter execute "$notebook" || exit 1; \
        else rm "$notebook"; fi; \
    done

EXPOSE 7860
# huggingface.co shows the Space in an iframe, which Jupyter's default CSP forbids.
# Voilà 0.5.13 leaks a kernel per page reload (its shutdown request fails the _xsrf
# check): culling reclaims the idle ones.
CMD ["voila", ".", "--port=7860", "--Voila.ip=0.0.0.0", "--no-browser", \
     "--Voila.tornado_settings={\"headers\": {\"Content-Security-Policy\": \"frame-ancestors 'self' https://huggingface.co\"}}", \
     "--MappingKernelManager.cull_interval=60", "--MappingKernelManager.cull_idle_timeout=300"]
