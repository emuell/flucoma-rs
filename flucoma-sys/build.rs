use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// -------------------------------------------------------------------------------------------------

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=patches/");
    println!("cargo:rerun-if-changed=../vendor/flucoma-core/include/");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let flucoma_dir = manifest_dir.join("..").join("vendor").join("flucoma-core");

    // -- apply patches to a copy of flucoma-core's headers

    let flucoma_dir_patched = out_dir.join("flucoma-core");
    if flucoma_dir_patched.exists() {
        fs::remove_dir_all(&flucoma_dir_patched)
            .expect("failed to clear patched flucoma-core headers");
    }
    copy_dir(
        &flucoma_dir.join("include"),
        &flucoma_dir_patched.join("include"),
    );
    apply_patches(&manifest_dir.join("patches"), &flucoma_dir_patched);

    // -- cmake configure + build ALL_BUILD

    let profile = match std::env::var("PROFILE").as_deref() {
        Ok("release") => "Release",
        _ => "RelWithDebInfo",
    };

    let mut cmake_config = cmake::Config::new(&flucoma_dir);
    cmake_config
        .profile(profile)
        .define("FOONATHAN_MEMORY_BUILD_TOOLS", "OFF")
        .define("FOONATHAN_MEMORY_BUILD_EXAMPLES", "OFF")
        .define("FOONATHAN_MEMORY_BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("FLUCOMA_TESTS", "OFF")
        .define("FMT_INSTALL", "OFF");
    if cfg!(target_env = "msvc") {
        cmake_config.define(
            // Use a msvc runtime library which is compatible with default Rust compiler settings
            "CMAKE_MSVC_RUNTIME_LIBRARY",
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL",
        );
        // Enable C++ exception handling (required by foonathan/memory)
        cmake_config.cxxflag("/EHsc");
    } else {
        // Disable all warnings
        cmake_config.cxxflag("-w");
    }

    let cmake_out = cmake_config.build();

    let cmake_build = cmake_out.join("build");
    let deps_dir = cmake_build.join("_deps");

    // -- configure and build foonathan_memory (required dependency)

    let memory_build_dir = deps_dir.join("memory-build");
    build_cmake_target(&memory_build_dir, "foonathan_memory", profile);

    // Locate and link the foonathan_memory static library.
    // The filename is versioned so we scan the directory to find the exact stem.
    let (memory_lib_dir, memory_lib_stem) =
        find_lib(&memory_build_dir.join("src"), profile, "foonathan_memory")
            .or_else(|| find_lib(&memory_build_dir, profile, "foonathan_memory"))
            .expect("could not find foonathan_memory library");
    println!("cargo:rustc-link-search=all={}", memory_lib_dir.display());
    println!("cargo:rustc-link-lib=static={}", memory_lib_stem);

    // -- Add system lib dependencies

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // -- Compile cpp! macro blocks via cpp_build

    let eigen_include = deps_dir.join("eigen-src");
    let hiss_include = deps_dir.join("hisstools-src").join("include");
    let spectra_include = deps_dir.join("spectra-src").join("include");
    let json_include = deps_dir.join("json-src").join("include");
    let fmt_include = deps_dir.join("fmt-src").join("include");
    let memory_config_include = memory_build_dir.join("src"); // config_impl.hpp
    let memory_include = deps_dir
        .join("memory-src")
        .join("include")
        .join("foonathan");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .static_crt(false) // match flucoma-core settings
        .include(flucoma_dir_patched.join("include"))
        .include(&eigen_include)
        .include(&hiss_include)
        .include(&spectra_include)
        .include(&json_include)
        .include(&fmt_include)
        .include(&memory_include)
        .include(&memory_config_include)
        .define("EIGEN_MPL2_ONLY", "1")
        .define("FMT_HEADER_ONLY", "1")
        .define("NOMINMAX", None)
        .define("_USE_MATH_DEFINES", None);
    if cfg!(target_env = "msvc") {
        // Enable C++ exception handling (required by foonathan/memory)
        build.flag("/EHsc").flag("/bigobj");
    } else {
        // Ignore all warnings
        build.flag("-fpermissive").flag("-w");
    }

    // NB: add -std=c++17 via flag_if_supported to avoid that cpp_build appends a -std=c++11
    let mut config: cpp_build::Config = build.clone().into();
    config
        .flag_if_supported("/std:c++17")
        .flag_if_supported("-std=c++17")
        .build("src/lib.rs");
}

// -------------------------------------------------------------------------------------------------

/// Recursively copy the directory `from` to `to`.
fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap_or_else(|e| panic!("failed to create {}: {}", to.display(), e));
    for entry in fs::read_dir(from)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", from.display(), e))
        .flatten()
    {
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target)
                .unwrap_or_else(|e| panic!("failed to copy {}: {}", entry.path().display(), e));
        }
    }
}

/// Apply all `*.patch` files in `patches_dir`, in name order, to the files below `root`.
/// Each patch must change a single file, with git style `a/` and `b/` path prefixes.
fn apply_patches(patches_dir: &Path, root: &Path) {
    let mut patches: Vec<PathBuf> = fs::read_dir(patches_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", patches_dir.display(), e))
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "patch")
        })
        .collect();
    patches.sort();

    for patch_path in patches {
        let patch_text = fs::read_to_string(&patch_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", patch_path.display(), e));
        let patch = diffy::Patch::from_str(&patch_text)
            .unwrap_or_else(|e| panic!("failed to parse {}: {}", patch_path.display(), e));
        let target = patch
            .modified()
            .and_then(|name| name.strip_prefix("b/"))
            .map(|name| root.join(name))
            .unwrap_or_else(|| panic!("{} names no b/ target file", patch_path.display()));
        // Checkouts on Windows may have turned the headers' line endings into CRLF
        let original = fs::read_to_string(&target)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", target.display(), e))
            .replace("\r\n", "\n");
        let patched = diffy::apply(&original, &patch)
            .unwrap_or_else(|e| panic!("failed to apply {}: {}", patch_path.display(), e));
        fs::write(&target, patched)
            .unwrap_or_else(|e| panic!("failed to write {}: {}", target.display(), e));
    }
}

// -------------------------------------------------------------------------------------------------

/// Build the ALL_BUILD target inside a cmake sub-directory.
fn build_cmake_target(dir: &PathBuf, label: &str, profile: &str) {
    let mut cmd = Command::new("cmake");
    cmd.arg("--build")
        .arg(dir)
        .arg("--config")
        .arg(profile)
        .arg("--parallel");

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run cmake --build for {}: {}", label, e));
    if !status.success() {
        panic!(
            "cmake --build {} failed (exit code: {:?})",
            label,
            status.code()
        );
    }
}

// -------------------------------------------------------------------------------------------------

/// Find a static library whose filename contains `name` as stem.
fn find_lib(base: &PathBuf, profile: &str, name: &str) -> Option<(PathBuf, String)> {
    // Scan `dir` for a `.lib` or `.a` file whose name contains `name`.
    let lib_stem_in = |dir: &PathBuf, name: &str| -> Option<String> {
        fs::read_dir(dir).ok()?.flatten().find_map(|e| {
            let fname = e.file_name().to_string_lossy().to_string();
            if fname.contains(name) && (fname.ends_with(".a") || fname.ends_with(".lib")) {
                if fname.ends_with(".lib") {
                    // XXX.lib on windows
                    return fname.strip_suffix(".lib").map(String::from);
                } else {
                    // libXXX.a on unix
                    return fname
                        .strip_prefix("lib")
                        .and_then(|s| s.strip_suffix(".a"))
                        .map(String::from);
                }
            }
            None
        })
    };

    // Try config-specific sub-dirs first (MSVC multi-config generators)
    for sub in &[profile, "Release", "RelWithDebInfo", "Debug"] {
        let d = base.join(sub);
        if let Some(stem) = lib_stem_in(&d, name) {
            return Some((d, stem));
        }
    }
    // Fallback: directly in base (Unix Makefile / Ninja generators)
    if let Some(stem) = lib_stem_in(base, name) {
        return Some((base.clone(), stem));
    }
    None
}
