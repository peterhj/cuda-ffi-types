extern crate bindgen;

use std::env;
use std::fs;
use std::path::{PathBuf};

fn main() {
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
  let cuda_dir = PathBuf::from(match env::var("CUDA_HOME") {
    Ok(path) => path,
    Err(_) => "/usr/local/cuda".to_owned(),
  });

  fs::remove_file(out_dir.join("cuda.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
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
    .write_to_file(out_dir.join("cuda.rs"))
    .expect("bindgen failed to write driver bindings");

  fs::remove_file(out_dir.join("cuda_runtime_api.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrapped_cuda_runtime_api.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaDeviceProp")
    .whitelist_type("cudaStreamCallback_t")
    .generate()
    .expect("bindgen failed to generate runtime bindings")
    .write_to_file(out_dir.join("cuda_runtime_api.rs"))
    .expect("bindgen failed to write runtime bindings");

  fs::remove_file(out_dir.join("driver_types.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
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
    .write_to_file(out_dir.join("driver_types.rs"))
    .expect("bindgen failed to write driver types bindings");

  fs::remove_file(out_dir.join("library_types.rs")).ok();
  bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrapped_library_types.h")
    .whitelist_recursively(false)
    .whitelist_type("cudaDataType")
    .whitelist_type("cudaDataType_t")
    .generate()
    .expect("bindgen failed to generate library types bindings")
    .write_to_file(out_dir.join("library_types.rs"))
    .expect("bindgen failed to write library types bindings");
}
