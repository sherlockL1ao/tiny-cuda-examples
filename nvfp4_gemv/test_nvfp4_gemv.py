from nvfp4_gemv_inline import nvfp4_gemv_cuda
from nvfp4_gemv_reference import generate_input, ref_kernel
from utils import make_match_reference

data = generate_input(m=256, k=512, l=10, seed=42)

print(data)
ref_out = ref_kernel(data)

check_impl = make_match_reference(nvfp4_gemv_cuda, rtol=1e-03, atol=1e-03)
check_impl(data, ref_out)

