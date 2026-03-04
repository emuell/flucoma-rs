use flucoma_sys as sys;
// isize and FlucomaIndex are technically identical
// prefer using FlucomaIndex to explicitly imply FFI usage
use flucoma_sys::FlucomaIndex;

pub struct KDTree {
    inner: *mut u8,
    dims: FlucomaIndex,
}

pub struct KNNResult {
    pub distances: Vec<f64>,
    pub ids: Vec<String>,
}

impl KDTree {
    pub fn new(dims: usize) -> Self {
        Self {
            inner: sys::kdtree_create(dims as FlucomaIndex),
            dims: dims as FlucomaIndex,
        }
    }

    pub fn add(&mut self, id: &str, data: &[f64]) {
        assert_eq!(
            data.len() as FlucomaIndex,
            self.dims,
            "Input dimensions ({}) do not match KDTree dimensions ({})",
            data.len(),
            self.dims
        );
        let c_id = std::ffi::CString::new(id).expect("CString::new failed");
        sys::kdtree_add_node(
            self.inner,
            c_id.as_ptr() as *const u8,
            data.as_ptr(),
            data.len() as FlucomaIndex,
        );
    }

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

        sys::kdtree_k_nearest(
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
        sys::kdtree_destroy(self.inner);
    }
}

// SAFETY: flucoma algorithms are thread-safe to move between threads.
unsafe impl Send for KDTree {}

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
