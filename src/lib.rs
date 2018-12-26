#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod cuda {
include!(concat!(env!("OUT_DIR"), "/cuda.rs"));
}

pub mod cuda_runtime_api {
use crate::driver_types::*;
include!(concat!(env!("OUT_DIR"), "/cuda_runtime_api.rs"));
}

pub mod driver_types {
use crate::cuda::*;
include!(concat!(env!("OUT_DIR"), "/driver_types.rs"));
}

pub mod library_types {
include!(concat!(env!("OUT_DIR"), "/library_types.rs"));
}
