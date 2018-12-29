#[cfg(feature = "fresh")]
extern crate bindgen;

#[cfg(feature = "fresh")]
use std::env;
#[cfg(feature = "fresh")]
use std::fs;
#[cfg(feature = "fresh")]
use std::path::{PathBuf};

#[cfg(all(
    not(feature = "fresh"),
    any(
        feature = "cuda_6_5",
        feature = "cuda_7_0",
        feature = "cuda_7_5",
        feature = "cuda_8_0",
        feature = "cuda_9_0",
        feature = "cuda_9_1",
        feature = "cuda_9_2",
        feature = "cuda_10_0",
    )
))]
fn main() {}

#[cfg(all(
    not(feature = "fresh"),
    not(any(
        feature = "cuda_6_5",
        feature = "cuda_7_0",
        feature = "cuda_7_5",
        feature = "cuda_8_0",
        feature = "cuda_9_0",
        feature = "cuda_9_1",
        feature = "cuda_9_2",
        feature = "cuda_10_0",
    ))
))]
fn main() {
  compile_error!("a cuda version feature must be enabled");
}

#[cfg(feature = "fresh")]
fn main() {
  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let cuda_dir = PathBuf::from(
      env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_owned())
  );
  let cuda_include_dir = cuda_dir.join("include");

  #[cfg(feature = "cuda_6_5")]
  let a_cuda_version_feature_must_be_enabled = "v6_5";
  #[cfg(feature = "cuda_7_0")]
  let a_cuda_version_feature_must_be_enabled = "v7_0";
  #[cfg(feature = "cuda_7_5")]
  let a_cuda_version_feature_must_be_enabled = "v7_5";
  #[cfg(feature = "cuda_8_0")]
  let a_cuda_version_feature_must_be_enabled = "v8_0";
  #[cfg(feature = "cuda_9_0")]
  let a_cuda_version_feature_must_be_enabled = "v9_0";
  #[cfg(feature = "cuda_9_1")]
  let a_cuda_version_feature_must_be_enabled = "v9_1";
  #[cfg(feature = "cuda_9_2")]
  let a_cuda_version_feature_must_be_enabled = "v9_2";
  #[cfg(feature = "cuda_10_0")]
  let a_cuda_version_feature_must_be_enabled = "v10_0";
  let v = a_cuda_version_feature_must_be_enabled;

  let gensrc_dir = manifest_dir.join("src").join(v);
  fs::create_dir(&gensrc_dir).ok();

  fs::remove_file(gensrc_dir.join("_cuda.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_include_dir.as_os_str().to_str().unwrap()))
    .header("wrapped_cuda.h")
    .whitelist_recursively(false)
    .whitelist_var("__CUDA_API_VERSION")
    .whitelist_var("CUDA_VERSION")
    .whitelist_type("cudaError_enum")
    .whitelist_type("CUresult")
    .whitelist_type("CUdevice")
    .whitelist_type("CUdevice_attribute")
    .whitelist_type("CUdevice_attribute_enum")
    .whitelist_type("CUuuid_st")
    .whitelist_type("CUuuid")
    .whitelist_type("CUctx_st")
    .whitelist_type("CUcontext")
    .whitelist_type("CUstream_st")
    .whitelist_type("CUstream")
    .whitelist_type("CUstreamCallback")
    .whitelist_type("CUevent_st")
    .whitelist_type("CUevent")
    .whitelist_type("CUdeviceptr")
    .whitelist_type("CUjit_option_enum")
    .whitelist_type("CUjit_option")
    .whitelist_type("CUmod_st")
    .whitelist_type("CUmodule")
    .whitelist_type("CUhostFn")
    .whitelist_type("CUfunc_st")
    .whitelist_type("CUfunction")
    .whitelist_type("CUDA_LAUNCH_PARAMS_st")
    .whitelist_type("CUDA_LAUNCH_PARAMS")
    .generate()
    .expect("bindgen failed to generate driver bindings")
    .write_to_file(gensrc_dir.join("_cuda.rs"))
    .expect("bindgen failed to write driver bindings");

  fs::remove_file(gensrc_dir.join("_cuda_fp16.rs")).ok();
  #[cfg(feature = "cuda_gte_9_0")]
  {
    bindgen::Builder::default()
      .clang_arg(format!("-I{}", cuda_include_dir.as_os_str().to_str().unwrap()))
      .clang_arg("-x").clang_arg("c++")
      .clang_arg("-std=c++11")
      .header("wrapped_cuda_fp16.h")
      .whitelist_recursively(false)
      .whitelist_type("__half")
      .whitelist_type("__half2")
      .generate()
      .expect("bindgen failed to generate fp16 bindings")
      .write_to_file(gensrc_dir.join("_cuda_fp16.rs"))
      .expect("bindgen failed to write fp16 bindings");
  }

  fs::remove_file(gensrc_dir.join("_cuda_runtime_api.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_include_dir.as_os_str().to_str().unwrap()))
    .header("wrapped_cuda_runtime_api.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaDeviceProp")
    .whitelist_type("cudaStreamCallback_t")
    .generate()
    .expect("bindgen failed to generate runtime bindings")
    .write_to_file(gensrc_dir.join("_cuda_runtime_api.rs"))
    .expect("bindgen failed to write runtime bindings");

  fs::remove_file(gensrc_dir.join("_driver_types.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_include_dir.as_os_str().to_str().unwrap()))
    .header("wrapped_driver_types.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaError")
    .whitelist_type("cudaError_t")
    .whitelist_type("cudaDeviceAttr")
    .whitelist_type("cudaStream_t")
    .whitelist_type("cudaEvent_t")
    .whitelist_type("cudaMemoryAdvise")
    .whitelist_type("cudaMemcpyKind")
    .whitelist_type("cudaMemRangeAttribute")
    .whitelist_type("cudaGLDeviceList")
    .whitelist_type("cudaGraphicsResource")
    .whitelist_type("cudaGraphicsResource_t")
    .generate()
    .expect("bindgen failed to generate driver types bindings")
    .write_to_file(gensrc_dir.join("_driver_types.rs"))
    .expect("bindgen failed to write driver types bindings");

  fs::remove_file(gensrc_dir.join("_library_types.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_include_dir.as_os_str().to_str().unwrap()))
    .header("wrapped_library_types.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaDataType")
    .whitelist_type("cudaDataType_t")
    .generate()
    .expect("bindgen failed to generate library types bindings")
    .write_to_file(gensrc_dir.join("_library_types.rs"))
    .expect("bindgen failed to write library types bindings");
}
