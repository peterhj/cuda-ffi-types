/* automatically generated by rust-bindgen */

pub type cudaStreamCallback_t = ::std::option::Option<
    unsafe extern "C" fn(
        stream: cudaStream_t,
        status: cudaError_t,
        userData: *mut ::std::os::raw::c_void,
    ),
>;
