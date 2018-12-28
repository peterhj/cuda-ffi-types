#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#[cfg(feature = "cuda_version")]
#[macro_use] extern crate static_assertions;

pub mod cuda {
include!(concat!(env!("OUT_DIR"), "/cuda.rs"));
}

#[cfg(feature = "gte_cuda_8_0")]
pub mod cuda_fp16 {
include!(concat!(env!("OUT_DIR"), "/cuda_fp16.rs"));
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

mod version_checks {
  #[cfg(feature = "cuda_6_5")]  const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION,  6050);
  #[cfg(feature = "cuda_6_5")]  const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,        6050);

  #[cfg(feature = "cuda_7_0")]  const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION,  7000);
  #[cfg(feature = "cuda_7_0")]  const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,        7000);

  #[cfg(feature = "cuda_7_5")]  const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION,  7050);
  #[cfg(feature = "cuda_7_5")]  const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,        7050);

  #[cfg(feature = "cuda_8_0")]  const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION,  8000);
  #[cfg(feature = "cuda_8_0")]  const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,        8000);

  #[cfg(feature = "cuda_9_0")]  const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION,  9000);
  #[cfg(feature = "cuda_9_0")]  const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,        9000);

  #[cfg(feature = "cuda_9_1")]  const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION,  9010);
  #[cfg(feature = "cuda_9_1")]  const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,        9010);

  #[cfg(feature = "cuda_9_2")]  const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION,  9020);
  #[cfg(feature = "cuda_9_2")]  const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,        9020);

  #[cfg(feature = "cuda_10_0")] const_assert_eq!(cuda_api_version; crate::cuda::__CUDA_API_VERSION, 10000);
  #[cfg(feature = "cuda_10_0")] const_assert_eq!(cuda_version;     crate::cuda::CUDA_VERSION,       10000);
}
