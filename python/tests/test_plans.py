# Licensed to the Apache Software Foundation (ASF) under one
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
import pyarrow

from datafusion import SessionContext, LogicalPlan, ExecutionPlan, DataFrame
import pytest


# Note: We must use CSV because memory tables are currently not supported for
# conversion to/from protobuf.
@pytest.fixture
def df() -> DataFrame:
    ctx = SessionContext()
    return ctx.read_csv(path="testing/data/csv/aggregate_test_100.csv").select("c1")


def test_logical_plan_to_proto(ctx, df) -> None:
    logical_plan_bytes = df.logical_plan().to_proto()
    logical_plan = LogicalPlan.from_proto(ctx, logical_plan_bytes)

    df_round_trip = ctx.create_dataframe_from_logical_plan(logical_plan)

    assert df.collect() == df_round_trip.collect()

    original_execution_plan = df.execution_plan()
    execution_plan_bytes = original_execution_plan.to_proto()
    execution_plan = ExecutionPlan.from_proto(ctx, execution_plan_bytes)

    assert str(original_execution_plan) == str(execution_plan)


def test_logical_plan_parameters(ctx) -> None:
    ctx.register_csv("t", path="testing/data/csv/aggregate_test_100.csv")
    plan = ctx.sql("SELECT c1, c2, c3 FROM t WHERE c1 = $1 AND c2 < $2").logical_plan()

    schema = plan.schema()
    assert schema.names == ["c1", "c2", "c3"]
    assert schema.field(0).type == pyarrow.string()
    assert schema.field(1).type == pyarrow.int64()
    assert schema.field(2).type == pyarrow.int64()

    parameters = plan.parameters()
    assert len(parameters) == 2
    assert parameters["$1"] == pyarrow.string()
    assert parameters["$2"] == pyarrow.int64()


def test_logical_plan_bind_positional_parameters(ctx) -> None:
    ctx.register_csv("t", path="testing/data/csv/aggregate_test_100.csv")
    plan = ctx.sql(
        "SELECT c1, c2, c4 FROM t WHERE c1 != $1 AND c2 > $2 and c4 < $3"
    ).logical_plan()

    new_plan = plan.with_parameter_values(param_values=["c", 3, -30000])
    result = ctx.execute_logical_plan(new_plan)

    result.show()
    assert result.count() == 2


def test_logical_plan_bind_named_parameters(ctx) -> None:
    ctx.register_csv("t", path="testing/data/csv/aggregate_test_100.csv")
    plan = ctx.sql(
        "SELECT c1, c2, c4 FROM t WHERE c1 != $str AND c2 > $int1 and c4 < $int2"
    ).logical_plan()

    new_plan = plan.with_parameter_values(param_values=["c", 3, -30000])
    result = ctx.execute_logical_plan(new_plan)

    result.show()
    assert result.count() == 2
