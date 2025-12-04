from nvfp4_gemv_inline import nvfp4_gemv_cuda
from nvfp4_gemv_reference import generate_input, ref_kernel
from utils import make_match_reference

data = generate_input(m=128, k=512, l=4, seed=42)
ref_out = ref_kernel(data)

data = generate_input(m=128, k=512, l=4, seed=42)
check_impl = make_match_reference(nvfp4_gemv_cuda, rtol=1e-03, atol=1e-03)
matched, msg = check_impl(data, ref_out)

if not matched: print(msg)
