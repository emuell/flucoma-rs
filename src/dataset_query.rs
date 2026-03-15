use flucoma_sys::dataset_query_process;

#[derive(Debug, Clone, Copy)]
#[repr(isize)]
pub enum ComparisonOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

#[derive(Debug, Clone, Copy)]
pub struct QueryCondition {
    pub column: usize,
    pub op: ComparisonOp,
    pub value: f64,
    /// `true` = AND condition list, `false` = OR condition list.
    pub and_group: bool,
}

#[derive(Debug, Clone)]
pub struct DataSetQueryResult {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub source_indices: Vec<usize>,
}

pub struct DataSetQuery;

impl DataSetQuery {
    pub fn execute(
        data: &[f64],
        rows: usize,
        cols: usize,
        selected_columns: &[usize],
        conditions: &[QueryCondition],
        limit: Option<usize>,
    ) -> Result<DataSetQueryResult, &'static str> {
        if rows == 0 || cols == 0 {
            return Err("rows and cols must be > 0");
        }
        if data.len() != rows * cols {
            return Err("data length does not match rows * cols");
        }
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
            data.as_ptr(),
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
            data: out_data,
            rows: count,
            cols: selected_columns.len(),
            source_indices: out_ids.into_iter().map(|x| x as usize).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_select_and_filter() {
        // 5x3 row-major
        let data = vec![
            0.0, 10.0, 100.0, //
            1.0, 20.0, 200.0, //
            2.0, 30.0, 300.0, //
            3.0, 40.0, 400.0, //
            4.0, 50.0, 500.0,
        ];
        let conditions = [QueryCondition {
            column: 0,
            op: ComparisonOp::Ge,
            value: 2.0,
            and_group: true,
        }];
        let res = DataSetQuery::execute(&data, 5, 3, &[1, 2], &conditions, Some(2)).unwrap();
        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res.data.len(), 4);
        assert_eq!(res.source_indices.len(), 2);
    }
}
