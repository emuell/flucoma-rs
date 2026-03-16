use flucoma_sys::{kdtree_add_node, kdtree_create, kdtree_destroy, kdtree_k_nearest, FlucomaIndex};

// -------------------------------------------------------------------------------------------------

/// Result of a [`KDTree::k_nearest`] query.
pub struct KNNResult {
    /// Euclidean distances to each returned neighbour, in ascending order.
    pub distances: Vec<f64>,
    /// String identifiers of the returned neighbours, in ascending distance order.
    pub ids: Vec<String>,
}

// -------------------------------------------------------------------------------------------------

/// K-d tree for fast nearest-neighbour search over fixed-dimensional point sets.
///
/// Points are identified by string labels and stored in a balanced k-d tree.
/// After adding points with [`add`](KDTree::add), call
/// [`k_nearest`](KDTree::k_nearest) to retrieve the closest neighbours to a
/// query vector.
///
/// # Usage
/// ```no_run
/// use flucoma_rs::data::KDTree;
///
/// let mut tree = KDTree::new(3);
/// tree.add("a", &[1.0, 0.0, 0.0]);
/// tree.add("b", &[0.0, 1.0, 0.0]);
/// tree.add("c", &[0.0, 0.0, 1.0]);
///
/// let result = tree.k_nearest(&[0.9, 0.1, 0.0], 2);
/// println!("nearest: {:?}", result.ids);
/// ```
///
/// See <https://learn.flucoma.org/reference/kdtree>
pub struct KDTree {
    inner: *mut u8,
    dims: FlucomaIndex,
}

unsafe impl Send for KDTree {}

impl KDTree {
    /// Create a new k-d tree for points with `dims` dimensions.
    pub fn new(dims: usize) -> Self {
        Self {
            inner: kdtree_create(dims as FlucomaIndex),
            dims: dims as FlucomaIndex,
        }
    }

    /// Add a labelled point to the tree.
    ///
    /// # Arguments
    /// * `id` — string label for the point.
    /// * `data` — feature vector; length must equal `dims`.
    ///
    /// # Panics
    /// Panics if `data.len() != dims`.
    pub fn add(&mut self, id: &str, data: &[f64]) {
        assert_eq!(
            data.len() as FlucomaIndex,
            self.dims,
            "Input dimensions ({}) do not match KDTree dimensions ({})",
            data.len(),
            self.dims
        );
        let c_id = std::ffi::CString::new(id).expect("CString::new failed");
        kdtree_add_node(
            self.inner,
            c_id.as_ptr() as *const u8,
            data.as_ptr(),
            data.len() as FlucomaIndex,
        );
    }

    /// Find the `k` nearest neighbours to `input`.
    ///
    /// Returns a `KNNResult` with up to `k` entries sorted by ascending
    /// Euclidean distance. Fewer than `k` entries are returned when the tree
    /// contains fewer than `k` points.
    ///
    /// # Arguments
    /// * `input` — query vector; length must equal `dims`.
    /// * `k` — maximum number of neighbours to return.
    ///
    /// # Panics
    /// Panics if `input.len() != dims`.
    pub fn k_nearest(&self, input: &[f64], k: usize) -> KNNResult {
        assert_eq!(
            input.len() as FlucomaIndex,
            self.dims,
            "Input dimensions ({}) do not match KDTree dimensions ({})",
            input.len(),
            self.dims
        );
        let mut distances = vec![0.0; k];
        let mut id_ptrs = vec![std::ptr::null::<u8>(); k];

        kdtree_k_nearest(
            self.inner,
            input.as_ptr(),
            input.len() as FlucomaIndex,
            k as FlucomaIndex,
            0.0,
            distances.as_mut_ptr(),
            id_ptrs.as_mut_ptr(),
        );

        let ids: Vec<String> = id_ptrs
            .into_iter()
            .take_while(|&p| !p.is_null())
            .map(|p| unsafe {
                std::ffi::CStr::from_ptr(p as *const std::os::raw::c_char)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        // Shrink distances to match actual returned IDs count
        distances.truncate(ids.len());

        KNNResult { distances, ids }
    }
}

impl Drop for KDTree {
    fn drop(&mut self) {
        kdtree_destroy(self.inner);
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kdtree_add_and_search() {
        let mut tree = KDTree::new(2);
        tree.add("origin", &[0.0, 0.0]);
        tree.add("right", &[10.0, 0.0]);
        tree.add("up", &[0.0, 10.0]);
        tree.add("diagonal", &[7.0, 7.0]);

        let target = [1.0, 1.0];
        let result = tree.k_nearest(&target, 2);
        assert_eq!(result.ids.len(), 2);
        assert_eq!(result.ids[0], "origin");

        let target2 = [8.0, 2.0];
        let result2 = tree.k_nearest(&target2, 1);
        assert_eq!(result2.ids.len(), 1);
        // [8.0, 2.0] is distance sqrt((8-10)^2 + (2-0)^2) = sqrt(4+4) = sqrt(8) to "right"
        // [8.0, 2.0] is distance sqrt((8-7)^2 + (2-7)^2) = sqrt(1+25) = sqrt(26) to "diagonal"
        assert_eq!(result2.ids[0], "right");
    }
}
