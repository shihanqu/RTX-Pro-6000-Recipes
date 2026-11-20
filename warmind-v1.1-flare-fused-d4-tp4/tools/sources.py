"""Materialize the immutable public BRRR D4 build kit; never use a private image as a base."""

import hashlib
import subprocess
import tempfile
from pathlib import Path

KIT_REPO = "https://github.com/shihanqu/RTX-Pro-6000-Recipes.git"
KIT_REV = "03e7a8a7ff48607d8fc7c155c1e965948cab5c54"
KIT_DIR = "deepseek-v41-flare-brrr-d4-tp4"
FLARE_REPO = "https://github.com/NVIDIA/hyperflare-inference.git"
FLARE_REV = "972e28e44781d8d0c84d70ae434db92c36076a43"
FLASHINFER_REV = "1f8ed8a3c46e27d03cff54e08f23e9bc50476d57"
REUSE_REV = "e16f949554f4efe8ee4154165231633e1cad8c05"
ASSETS = {
    "patches/flare-brrr-d4.patch": "85a2cafdf4e957afa1cac67a23e8ee2e50009c55dbfc1797a78adbef5cf12c79",
    "patches/flashinfer-route-reuse.patch": "285d8eb04790d39ec26a42b8f1da0329c9b85a19b5b3639800205bd1e37b2318",
    "prepare/Dockerfile": "e014127d30442ef5318f567d67cca0a82480f1e05d36fd2edc6eeb8a38542d06",
    "prepare/Dockerfile.dockerignore": "92074e935b7a0b2040eb2a3e8c8506efecca90b099c5014c619878c5f5268533",
}


def fetch(work, prefix, url, revision):
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=work))
    git = ["git", "-C", str(path)]
    subprocess.run([*git, "init", "--quiet"], check=True)
    subprocess.run([*git, "fetch", "--quiet", "--depth", "1", url, revision], check=True)
    if subprocess.check_output([*git, "rev-parse", "FETCH_HEAD"], text=True).strip() != revision:
        raise ValueError("Unexpected Git revision")
    subprocess.run([*git, "switch", "--quiet", "--detach", revision], check=True)
    return path


def prepare(recipe, work):
    git = ["git", "-C", str(recipe)]
    if subprocess.run(
        [*git, "cat-file", "-e", f"{KIT_REV}^{{commit}}"], stderr=subprocess.DEVNULL
    ).returncode:
        git = ["git", "-C", str(fetch(work, "build-kit-", KIT_REPO, KIT_REV))]
    assets = {}
    for name, digest in ASSETS.items():
        data = subprocess.check_output([*git, "cat-file", "blob", f"{KIT_REV}:{KIT_DIR}/{name}"])
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Invalid build asset: {name}")
        assets[name] = data
    src = fetch(work, "flare-source-", FLARE_REPO, FLARE_REV)
    patch = assets["patches/flare-brrr-d4.patch"]
    for options in (["--check"], []):
        subprocess.run(["git", "-C", str(src), "apply", *options, "-"], input=patch, check=True)
    for source, dest in [
        ("prepare/Dockerfile", "Dockerfile.recipe"),
        ("prepare/Dockerfile.dockerignore", "Dockerfile.recipe.dockerignore"),
        ("patches/flashinfer-route-reuse.patch", "flashinfer-route-reuse.patch"),
    ]:
        (src / dest).write_bytes(assets[source])
    return src
