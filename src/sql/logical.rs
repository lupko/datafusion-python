// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::expr::aggregate::PyAggregate;
use crate::expr::analyze::PyAnalyze;
use crate::expr::distinct::PyDistinct;
use crate::expr::empty_relation::PyEmptyRelation;
use crate::expr::explain::PyExplain;
use crate::expr::extension::PyExtension;
use crate::expr::filter::PyFilter;
use crate::expr::join::PyJoin;
use crate::expr::limit::PyLimit;
use crate::expr::logical_node::LogicalNode;
use crate::expr::projection::PyProjection;
use crate::expr::sort::PySort;
use crate::expr::subquery::PySubquery;
use crate::expr::subquery_alias::PySubqueryAlias;
use crate::expr::table_scan::PyTableScan;
use crate::expr::unnest::PyUnnest;
use crate::expr::window::PyWindowExpr;
use crate::{context::PySessionContext, errors::py_unsupported_variant_err};
use arrow::datatypes::DataType;
use arrow::pyarrow::ToPyArrow;
use datafusion::common::{ParamValues, ScalarValue};
use datafusion::{error::DataFusionError, logical_expr::LogicalPlan};
use datafusion_proto::logical_plan::{AsLogicalPlan, DefaultLogicalExtensionCodec};
use prost::Message;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{IntoPyDict, PyDict};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyBytes};
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(name = "LogicalPlan", module = "datafusion", subclass)]
#[derive(Debug, Clone)]
pub struct PyLogicalPlan {
    pub(crate) plan: Arc<LogicalPlan>,
}

impl PyLogicalPlan {
    /// creates a new PyLogicalPlan
    pub fn new(plan: LogicalPlan) -> Self {
        Self {
            plan: Arc::new(plan),
        }
    }

    pub fn plan(&self) -> Arc<LogicalPlan> {
        self.plan.clone()
    }
}

macro_rules! extract_nullable_scalar {
    ($var:ident, $t:ty) => {
        if $var.is_none() {
            None
        } else {
            Some($var.extract::<$t>()?)
        }
    };
}

fn py_obj_to_scalar(
    py: Python<'_>,
    obj: &PyObject,
    data_type: &Option<DataType>,
) -> PyResult<ScalarValue> {
    if data_type.is_none() {
        // TODO: find out when can parameter data type end up None and
        //  whether it then makes sense to set parameter value to NULL..
        return Ok(ScalarValue::Null);
    }

    let bound = obj.bind(py);
    let value = match data_type.as_ref().unwrap() {
        DataType::Boolean => ScalarValue::Boolean(extract_nullable_scalar!(bound, bool)),
        DataType::Int16 => ScalarValue::Int16(extract_nullable_scalar!(bound, i16)),
        DataType::UInt16 => ScalarValue::UInt16(extract_nullable_scalar!(bound, u16)),
        DataType::Int32 => ScalarValue::Int32(extract_nullable_scalar!(bound, i32)),
        DataType::UInt32 => ScalarValue::UInt32(extract_nullable_scalar!(bound, u32)),
        DataType::Int64 => ScalarValue::Int64(extract_nullable_scalar!(bound, i64)),
        DataType::UInt64 => ScalarValue::UInt64(extract_nullable_scalar!(bound, u64)),
        DataType::Float32 => ScalarValue::Float32(extract_nullable_scalar!(bound, f32)),
        DataType::Float64 => ScalarValue::Float64(extract_nullable_scalar!(bound, f64)),
        DataType::Utf8 => ScalarValue::Utf8(extract_nullable_scalar!(bound, String)),
        DataType::LargeUtf8 => ScalarValue::Utf8(extract_nullable_scalar!(bound, String)),
        DataType::Binary => ScalarValue::Binary(extract_nullable_scalar!(bound, Vec<u8>)),
        DataType::LargeBinary => ScalarValue::LargeBinary(extract_nullable_scalar!(bound, Vec<u8>)),
        unsupported => {
            return Err(PyValueError::new_err(format!(
                "Unsupported parameter data type {}",
                unsupported
            )))
        }
    };

    Ok(value)
}

fn py_dict_to_param_values(
    py: Python<'_>,
    values: HashMap<String, PyObject>,
    parameter_types: &HashMap<String, Option<DataType>>,
) -> PyResult<ParamValues> {
    let mut result: HashMap<String, ScalarValue> = HashMap::default();

    for (param_id, obj) in values.iter() {
        let param_type = parameter_types.get(param_id).ok_or_else(|| {
            PyValueError::new_err(format!("Plan does not have named parameter {}", param_id))
        })?;

        let value = py_obj_to_scalar(py, obj, param_type).map_err(|e| {
            PyTypeError::new_err(format!(
                "Failed to convert value for parameter {}: {}",
                param_id,
                e.value_bound(py).str().unwrap()
            ))
        })?;

        result.insert(param_id.clone(), value);
    }

    Ok(result.into())
}

fn py_seq_to_param_values(
    py: Python<'_>,
    values: Vec<PyObject>,
    parameter_types: &HashMap<String, Option<DataType>>,
) -> PyResult<ParamValues> {
    let mut result: Vec<ScalarValue> = Vec::with_capacity(values.len());

    for (idx, obj) in values.iter().enumerate() {
        let param_id = format!("${}", idx + 1);
        let param_type = parameter_types.get(&param_id).ok_or_else(|| {
            PyValueError::new_err(format!("Plan does not have parameter {}", &param_id))
        })?;

        let value = py_obj_to_scalar(py, obj, param_type).map_err(|e| {
            PyTypeError::new_err(format!(
                "Failed to convert value for parameter {}: {}",
                param_id,
                e.value_bound(py).str().unwrap()
            ))
        })?;

        result.push(value);
    }

    Ok(result.into())
}

fn py_obj_to_param_values(
    values: Bound<'_, PyAny>,
    parameter_types: &HashMap<String, Option<DataType>>,
) -> PyResult<ParamValues> {
    if let Ok(seq) = values.extract::<Vec<PyObject>>() {
        py_seq_to_param_values(values.py(), seq, parameter_types)
    } else if let Ok(dict) = values.extract::<HashMap<String, PyObject>>() {
        py_dict_to_param_values(values.py(), dict, parameter_types)
    } else {
        Err(PyValueError::new_err(
            "Parameter values must be either a sequence (for positional parameters) or dictionary (for named parameters).",
        ))
    }
}

#[pymethods]
impl PyLogicalPlan {
    /// Return the specific logical operator
    pub fn to_variant(&self, py: Python) -> PyResult<PyObject> {
        match self.plan.as_ref() {
            LogicalPlan::Aggregate(plan) => PyAggregate::from(plan.clone()).to_variant(py),
            LogicalPlan::Analyze(plan) => PyAnalyze::from(plan.clone()).to_variant(py),
            LogicalPlan::Distinct(plan) => PyDistinct::from(plan.clone()).to_variant(py),
            LogicalPlan::EmptyRelation(plan) => PyEmptyRelation::from(plan.clone()).to_variant(py),
            LogicalPlan::Explain(plan) => PyExplain::from(plan.clone()).to_variant(py),
            LogicalPlan::Extension(plan) => PyExtension::from(plan.clone()).to_variant(py),
            LogicalPlan::Filter(plan) => PyFilter::from(plan.clone()).to_variant(py),
            LogicalPlan::Join(plan) => PyJoin::from(plan.clone()).to_variant(py),
            LogicalPlan::Limit(plan) => PyLimit::from(plan.clone()).to_variant(py),
            LogicalPlan::Projection(plan) => PyProjection::from(plan.clone()).to_variant(py),
            LogicalPlan::Sort(plan) => PySort::from(plan.clone()).to_variant(py),
            LogicalPlan::TableScan(plan) => PyTableScan::from(plan.clone()).to_variant(py),
            LogicalPlan::Subquery(plan) => PySubquery::from(plan.clone()).to_variant(py),
            LogicalPlan::SubqueryAlias(plan) => PySubqueryAlias::from(plan.clone()).to_variant(py),
            LogicalPlan::Unnest(plan) => PyUnnest::from(plan.clone()).to_variant(py),
            LogicalPlan::Window(plan) => PyWindowExpr::from(plan.clone()).to_variant(py),
            LogicalPlan::Repartition(_)
            | LogicalPlan::Union(_)
            | LogicalPlan::Statement(_)
            | LogicalPlan::Values(_)
            | LogicalPlan::Dml(_)
            | LogicalPlan::Ddl(_)
            | LogicalPlan::Copy(_)
            | LogicalPlan::DescribeTable(_)
            | LogicalPlan::RecursiveQuery(_) => Err(py_unsupported_variant_err(format!(
                "Conversion of variant not implemented: {:?}",
                self.plan
            ))),
        }
    }

    /// Get the inputs to this plan
    fn inputs(&self) -> Vec<PyLogicalPlan> {
        let mut inputs = vec![];
        for input in self.plan.inputs() {
            inputs.push(input.to_owned().into());
        }
        inputs
    }

    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let parameter_types: HashMap<String, Option<PyObject>> = self
            .plan
            .get_parameter_types()?
            .iter()
            .map(|(id, data_type)| {
                (
                    id.clone(),
                    data_type
                        .as_ref()
                        .map_or_else(|| None, |t| Some(t.to_pyarrow(py).unwrap())),
                )
            })
            .collect();

        Ok(parameter_types.into_py_dict_bound(py))
    }

    fn with_parameter_values(&self, param_values: Bound<'_, PyAny>) -> PyResult<Self> {
        let parameter_types = self.plan.get_parameter_types()?;
        let df_param_values = py_obj_to_param_values(param_values, &parameter_types)?;
        let plan = self
            .plan
            .as_ref()
            .clone()
            .with_param_values(df_param_values)?;

        Ok(Self::new(plan))
    }

    fn schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.plan.schema().inner().to_pyarrow(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.plan))
    }

    fn display(&self) -> String {
        format!("{}", self.plan.display())
    }

    fn display_indent(&self) -> String {
        format!("{}", self.plan.display_indent())
    }

    fn display_indent_schema(&self) -> String {
        format!("{}", self.plan.display_indent_schema())
    }

    fn display_graphviz(&self) -> String {
        format!("{}", self.plan.display_graphviz())
    }

    pub fn to_proto<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let codec = DefaultLogicalExtensionCodec {};
        let proto =
            datafusion_proto::protobuf::LogicalPlanNode::try_from_logical_plan(&self.plan, &codec)?;

        let bytes = proto.encode_to_vec();
        Ok(PyBytes::new_bound(py, &bytes))
    }

    #[staticmethod]
    pub fn from_proto(ctx: PySessionContext, proto_msg: Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes: &[u8] = proto_msg.extract()?;
        let proto_plan =
            datafusion_proto::protobuf::LogicalPlanNode::decode(bytes).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Unable to decode logical node from serialized bytes: {}",
                    e
                ))
            })?;

        let codec = DefaultLogicalExtensionCodec {};
        let plan = proto_plan
            .try_into_logical_plan(&ctx.ctx, &codec)
            .map_err(DataFusionError::from)?;
        Ok(Self::new(plan))
    }
}

impl From<PyLogicalPlan> for LogicalPlan {
    fn from(logical_plan: PyLogicalPlan) -> LogicalPlan {
        logical_plan.plan.as_ref().clone()
    }
}

impl From<LogicalPlan> for PyLogicalPlan {
    fn from(logical_plan: LogicalPlan) -> PyLogicalPlan {
        PyLogicalPlan {
            plan: Arc::new(logical_plan),
        }
    }
}
