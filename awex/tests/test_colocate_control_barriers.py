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

from unittest.mock import Mock, call

import pytest

from awex.writer import weights_writer as writer_module


@pytest.mark.parametrize("rank", [0, 1])
def test_colocate_control_barriers_reuse_host_group(monkeypatch, rank):
    """Control synchronization must not consume inference's device cores."""
    group = object()
    new_group = Mock(return_value=group)
    barrier = Mock()
    monkeypatch.setattr(writer_module.dist, "new_group", new_group)
    monkeypatch.setattr(writer_module.dist, "barrier", barrier)
    monkeypatch.setattr(writer_module.dist, "get_rank", lambda: rank)
    writer = object.__new__(writer_module.WeightsExchangeShardingWriter)
    writer.enable_colocate_mode = True
    writer.num_infer_engines = 2
    writer._colocate_control_group = None
    writer.timeout = 10
    writer.train_engine = Mock()
    writer.meta_server_client = Mock()

    for _ in range(2):
        writer._release_memory_for_weights_exchange()
        writer._finish_weights_update()

    new_group.assert_called_once_with(backend="gloo")
    assert barrier.call_args_list == [call(group=group)] * 4
    assert writer.meta_server_client.delete_if_exists.call_count == (
        6 if rank == 0 else 0
    )


def test_noncolocate_finish_does_not_create_control_group(monkeypatch):
    new_group, barrier = Mock(), Mock()
    monkeypatch.setattr(writer_module.dist, "new_group", new_group)
    monkeypatch.setattr(writer_module.dist, "barrier", barrier)
    writer = object.__new__(writer_module.WeightsExchangeShardingWriter)
    writer.enable_colocate_mode = False

    writer._finish_weights_update()

    new_group.assert_not_called()
    barrier.assert_not_called()
