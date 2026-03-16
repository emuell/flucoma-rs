use crate::matrix::Matrix;
use flucoma_sys::dataset_query_process;

// -------------------------------------------------------------------------------------------------

/// Comparison operator used in a [`DataSetQueryCondition`].
#[derive(Debug, Clone, Copy)]
#[repr(isize)]
pub enum DataSetComparisonOp {
    /// Equal (`==`).
    Eq = 0,
    /// Not equal (`!=`).
    Ne = 1,
    /// Less than (`<`).
    Lt = 2,
    /// Less than or equal (`<=`).
    Le = 3,
    /// Greater than (`>`).
    Gt = 4,
    /// Greater than or equal (`>=`).
    Ge = 5,
}

// -------------------------------------------------------------------------------------------------

/// A single filter condition applied to one column of a dataset in [`DataSetQuery`].
#[derive(Debug, Clone, Copy)]
pub struct DataSetQueryCondition {
    /// Zero-based column index to test.
    pub column: usize,
    /// Comparison operator.
    pub op: DataSetComparisonOp,
    /// Threshold value.
    pub value: f64,
    /// `true` — this condition is ANDed with others; `false` — ORed.
    pub and_group: bool,
}

// -------------------------------------------------------------------------------------------------

/// Result of a [`DataSetQuery::execute`] call.
#[derive(Debug, Clone)]
pub struct DataSetQueryResult {
    /// Row-major output matrix of the selected columns.
    pub data: Matrix,
    /// Row indices in the original dataset for each returned row.
    pub source_indices: Vec<usize>,
}

// -------------------------------------------------------------------------------------------------

/// SQL-like filter and column selection over a flat row-major dataset.
///
/// `DataSetQuery` lets you select a subset of columns and filter rows by one
/// or more conditions, optionally limiting the number of results.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::{DataSetQuery, DataSetQueryCondition, DataSetComparisonOp, Matrix};
///
/// // 4×3 row-major dataset
/// let data = Matrix::from_vec(vec![
///     0.0, 10.0, 100.0,
///     1.0, 20.0, 200.0,
///     2.0, 30.0, 300.0,
///     3.0, 40.0, 400.0,
/// ], 4, 3).unwrap();
///
/// let conditions = [DataSetQueryCondition {
///     column: 0,
///     op: DataSetComparisonOp::Ge,
///     value: 2.0,
///     and_group: true,
/// }];
///
/// let result = DataSetQuery::execute(&data, &[1, 2], &conditions, None).unwrap();
/// println!("{} rows matched", result.data.rows());
/// ```
///
/// See <https://learn.flucoma.org/reference/datasetquery>
pub struct DataSetQuery;

impl DataSetQuery {
    /// Execute a query against a row-major dataset.
    ///
    /// Selects `selected_columns` from rows that satisfy all `conditions`,
    /// returning at most `limit` rows (or all matching rows when `None`).
    ///
    /// # Arguments
    /// * `data` — row-major input matrix.
    /// * `selected_columns` — zero-based column indices to include in output (non-empty).
    /// * `conditions` — filter conditions; empty slice returns all rows.
    /// * `limit` — maximum number of rows to return, or `None` for no limit.
    ///
    /// # Errors
    /// Returns an error if `selected_columns` is empty, any column index is
    /// out of range, or the underlying query fails.
    pub fn execute(
        data: &Matrix,
        selected_columns: &[usize],
        conditions: &[DataSetQueryCondition],
        limit: Option<usize>,
    ) -> Result<DataSetQueryResult, &'static str> {
        let rows = data.rows();
        let cols = data.cols();

        if selected_columns.is_empty() {
            return Err("selected_columns cannot be empty");
        }
        if selected_columns.iter().any(|&c| c >= cols) {
            return Err("selected column out of range");
        }
        if conditions.iter().any(|c| c.column >= cols) {
            return Err("condition column out of range");
        }

        let selected_cols: Vec<isize> = selected_columns.iter().map(|&x| x as isize).collect();
        let cond_cols: Vec<isize> = conditions.iter().map(|c| c.column as isize).collect();
        let cond_ops: Vec<isize> = conditions.iter().map(|c| c.op as isize).collect();
        let cond_vals: Vec<f64> = conditions.iter().map(|c| c.value).collect();
        let cond_and: Vec<isize> = conditions
            .iter()
            .map(|c| if c.and_group { 1 } else { 0 })
            .collect();

        let mut out_data = vec![0.0; rows * selected_columns.len()];
        let mut out_ids = vec![0isize; rows];
        let mut out_count = 0isize;

        let ok = dataset_query_process(
            data.data().as_ptr(),
            rows as isize,
            cols as isize,
            selected_cols.as_ptr(),
            selected_cols.len() as isize,
            cond_cols.as_ptr(),
            cond_ops.as_ptr(),
            cond_vals.as_ptr(),
            cond_and.as_ptr(),
            cond_cols.len() as isize,
            limit.unwrap_or(0) as isize,
            out_data.as_mut_ptr(),
            out_ids.as_mut_ptr(),
            &mut out_count as *mut isize,
        );
        if !ok {
            return Err("dataset query failed");
        }

        let count = out_count as usize;
        out_data.truncate(count * selected_columns.len());
        out_ids.truncate(count);
        Ok(DataSetQueryResult {
            data: Matrix::from_vec(out_data, count, selected_columns.len()).unwrap(),
            source_indices: out_ids.into_iter().map(|x| x as usize).collect(),
        })
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_select_and_filter() {
        use crate::matrix::Matrix;
        // 5x3 row-major
        let data = Matrix::from_vec(
            vec![
                0.0, 10.0, 100.0, //
                1.0, 20.0, 200.0, //
                2.0, 30.0, 300.0, //
                3.0, 40.0, 400.0, //
                4.0, 50.0, 500.0,
            ],
            5,
            3,
        )
        .unwrap();
        let conditions = [DataSetQueryCondition {
            column: 0,
            op: DataSetComparisonOp::Ge,
            value: 2.0,
            and_group: true,
        }];
        let res = DataSetQuery::execute(&data, &[1, 2], &conditions, Some(2)).unwrap();
        assert_eq!(res.data.rows(), 2);
        assert_eq!(res.data.cols(), 2);
        assert_eq!(res.source_indices.len(), 2);
    }
}
