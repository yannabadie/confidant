"""The README's install path must give a working app.

A reviewer followed the README on 28 October 2026: `pip install -e ".[app]"` took the newest Gradio (6.x), and
`python app.py` stopped at `Blocks.launch() got an unexpected keyword argument 'show_api'`; `pip install -e
".[dev]"` then `pytest` failed nine tests for lack of Gradio. The pinned requirements.txt worked, so the Space did.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    tomllib = pytest.importorskip("tomli")


def extras() -> dict[str, list[str]]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def pinned_gradio() -> str:
    found = re.search(r"^gradio==([\d.]+)\s*$", (ROOT / "requirements.txt").read_text(encoding="utf-8"), re.M)
    assert found, "requirements.txt pins gradio for the Space"
    return found.group(1)


def gradio_spec(requirements: list[str]) -> str:
    specs = [r for r in requirements if re.match(r"gradio\b", r)]
    assert specs, f"no gradio in {requirements}"
    return specs[0]


@pytest.mark.parametrize("extra", ["app", "dev"])
def test_an_extra_that_runs_the_app_takes_a_gradio_it_works_with(extra: str) -> None:
    spec = gradio_spec(extras()[extra])
    assert "<6" in spec.replace(" ", ""), f"{extra} extra: {spec!r} would install Gradio 6, which app.py does not run on"
    lower = re.search(r">=\s*([\d.]+)", spec)
    assert lower, f"{extra} extra: {spec!r} has no lower bound"
    as_tuple = lambda version: tuple(int(part) for part in version.split("."))  # noqa: E731
    assert as_tuple(lower.group(1)) <= as_tuple(pinned_gradio()), "the Space's pinned Gradio must satisfy the extra"


def test_requirements_points_to_the_file_that_holds_the_sdk_version() -> None:
    comment = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()[1]
    named = re.findall(r"[\w/.-]+\.md", comment)
    assert named and all((ROOT / name).is_file() for name in named), comment
    card = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    assert f"sdk_version: {pinned_gradio()}" in card


def test_the_app_launches_as_app_py_launches_it() -> None:
    import app

    kwargs = dict(show_api=False, favicon_path=str(app.ASSETS / "logo.svg"))  # the same call as app.py's __main__
    demo = app.build_app()
    try:
        _, url, _ = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", server_port=None, quiet=True,
                                **kwargs)
        page = httpx.get(url, timeout=30, follow_redirects=True)
        assert page.status_code == 200 and "Confidant" in page.text
    finally:
        demo.close()
