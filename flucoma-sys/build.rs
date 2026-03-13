use std::fs;
use std::path::PathBuf;
use std::process::Command;

// -------------------------------------------------------------------------------------------------

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=../vendor/flucoma-core/include/");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let flucoma_dir = manifest_dir.join("..").join("vendor").join("flucoma-core");

    let profile = match std::env::var("PROFILE").as_deref() {
        Ok("release") => "Release",
        _ => "RelWithDebInfo",
    };

    // -- cmake configure + build ALL_BUILD

    let mut cmake_cfg = cmake::Config::new(&flucoma_dir);
    cmake_cfg
        .profile(profile)
        // Do not contact remotes to update already-populated FetchContent deps.
        // This keeps repeated CI/local builds deterministic and offline-safe.
        .define("FETCHCONTENT_UPDATES_DISCONNECTED", "ON")
        .define("FOONATHAN_MEMORY_BUILD_TOOLS", "OFF")
        .define("FOONATHAN_MEMORY_BUILD_EXAMPLES", "OFF")
        .define("FOONATHAN_MEMORY_BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("FLUCOMA_TESTS", "OFF")
        .define("FMT_INSTALL", "OFF")
        // Use /MT (static CRT) -- flucoma-core hardcodes this in its CMakeLists.txt
        .define(
            "CMAKE_MSVC_RUNTIME_LIBRARY",
            "MultiThreaded$<$<CONFIG:Debug>:Debug>",
        )
        // Enable C++ exception handling for msvc (required by foonathan/memory)
        .cxxflag(if cfg!(target_env = "msvc") {
            "/EHsc"
        } else {
            ""
        });

    // Optional: force fully offline configure (requires pre-populated sources).
    if env_truthy("FLUCOMA_FULLY_DISCONNECTED") {
        cmake_cfg.define("FETCHCONTENT_FULLY_DISCONNECTED", "ON");
    }

    // Optional: point dependencies to local source checkouts to support offline CI.
    apply_path_override(&mut cmake_cfg, "FLUCOMA_HISS_PATH", "HISS_PATH");
    apply_path_override(&mut cmake_cfg, "FLUCOMA_EIGEN_PATH", "EIGEN_PATH");
    apply_path_override(&mut cmake_cfg, "FLUCOMA_SPECTRA_PATH", "SPECTRA_PATH");
    apply_path_override(&mut cmake_cfg, "FLUCOMA_JSON_PATH", "JSON_PATH");
    apply_path_override(
        &mut cmake_cfg,
        "FLUCOMA_MEMORY_PATH",
        "FETCHCONTENT_SOURCE_DIR_MEMORY",
    );
    apply_path_override(
        &mut cmake_cfg,
        "FLUCOMA_FMT_PATH",
        "FETCHCONTENT_SOURCE_DIR_FMT",
    );

    let cmake_out = cmake_cfg.build();

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
        .static_crt(true) // see above /MT
        .include(flucoma_dir.join("include"))
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
        .define("_USE_MATH_DEFINES", None)
        .flag_if_supported("-Wno-unused");

    if cfg!(target_env = "msvc") {
        build.flag("/EHsc").flag("/bigobj");
    }
    build.flag_if_supported("-fpermissive");

    // NB: add -std=c++17 via flag_if_supported to avoid that cpp_build appends a -std=c++11
    let mut config: cpp_build::Config = build.clone().into();
    config
        .flag_if_supported("/std:c++17")
        .flag_if_supported("-std=c++17")
        .build("src/lib.rs");
}

// -------------------------------------------------------------------------------------------------

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "on" | "yes")
        })
        .unwrap_or(false)
}

fn apply_path_override(cfg: &mut cmake::Config, env_var: &str, cmake_var: &str) {
    if let Ok(path) = std::env::var(env_var) {
        let path = path.trim();
        if !path.is_empty() {
            cfg.define(cmake_var, path);
        }
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
