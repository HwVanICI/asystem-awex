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


class FSDPToHFWeightConverter:
    """Identity converter for AReaL FSDP parameters.

    AReaL exposes FSDP parameters to Awex after applying its rollout-engine name
    mapping, so the writer can transfer the local shard under the same HF key.
    """

    def __init__(self, hf_config, rank_info, infer_conf):
        self.hf_config = hf_config
        self.rank_info = rank_info
        self.infer_conf = infer_conf

    def convert_param(self, name, param, **kwargs):
        del kwargs
        return [(name, param)]
