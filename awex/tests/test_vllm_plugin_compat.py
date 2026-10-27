# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Exercise vLLM API layouts without importing its GPU runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel


def add_module(monkeypatch, name):
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        monkeypatch.setattr(
            add_module(monkeypatch, parent), child, module, raising=False
        )
    return module


@pytest.fixture
def load_plugin(monkeypatch):
    # Isolate the simulated layout even when real vLLM is installed.
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name)
    add_module(monkeypatch, "vllm")

    def load(layout):
        protocol_path = {
            "legacy": "vllm.entrypoints.openai.protocol",
            "split": "vllm.entrypoints.openai.engine.protocol",
            "serve": "vllm.entrypoints.serve.engine.protocol",
        }[layout]
        add_module(monkeypatch, protocol_path).OpenAIBaseModel = BaseModel
        server_path = (
            "vllm.entrypoints.launchers.api_server.entry"
            if layout == "serve"
            else "vllm.entrypoints.openai.api_server"
        )
        server = add_module(monkeypatch, server_path)
        if layout == "legacy":
            server.router = APIRouter()

        def build_app(*args, **kwargs):
            app = FastAPI()
            app.state.build_args = (args, kwargs)
            if layout == "legacy":
                app.include_router(server.router)
            return app

        server.build_app = build_app
        path = Path(__file__).parents[1] / "vllm_plugin.py"
        spec = importlib.util.spec_from_file_location("_awex_plugin_compat", path)
        plugin = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, plugin)
        spec.loader.exec_module(plugin)
        monkeypatch.setattr(plugin, "_patch_awex_worker", lambda: None)
        return plugin, server, build_app

    return load


@pytest.mark.parametrize("layout", ["legacy", "split", "serve"])
def test_routes_attach_once_and_build_app_arguments_survive(load_plugin, layout):
    plugin, server, original = load_plugin(layout)
    assert plugin.OpenAIBaseModel is BaseModel
    plugin.register_awex_plugin()
    installed = server.build_app
    plugin.register_awex_plugin()
    assert server.build_app is installed
    assert (installed is original) == (layout == "legacy")
    expected_paths = [route.path for route in plugin.router.routes]
    assert "/areal_awex_init" in expected_paths
    for _ in range(2):
        app = server.build_app("args", supported_tasks=("generate",))
        assert app.state.build_args == (("args",), {"supported_tasks": ("generate",)})
        paths = [route.path for route in app.routes]
        for path in expected_paths:
            assert paths.count(path) == 1


def test_new_protocol_takes_precedence_over_older_paths(load_plugin, monkeypatch):
    class LegacyBaseModel(BaseModel):
        pass

    add_module(
        monkeypatch, "vllm.entrypoints.openai.engine.protocol"
    ).OpenAIBaseModel = LegacyBaseModel
    add_module(
        monkeypatch, "vllm.entrypoints.openai.protocol"
    ).OpenAIBaseModel = LegacyBaseModel
    plugin, _, _ = load_plugin("serve")
    assert plugin.OpenAIBaseModel is BaseModel
