#include <torch/all.h>
#include <torch/python.h>

void vecquant2matmul_sycl(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant2matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  vecquant2matmul_sycl(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant3matmul_sycl(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  vecquant3matmul_sycl(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant4matmul_sycl(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  vecquant4matmul_sycl(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant8matmul_sycl(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  vecquant8matmul_sycl(vec, mat, mul, scales, zeros, g_idx);
}


// old

void vecquant2matmul_sycl_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant2matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  vecquant2matmul_sycl_old(vec, mat, mul, scales, zeros,groupsize);
}

void vecquant3matmul_sycl_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant3matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  vecquant3matmul_sycl_old(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant4matmul_sycl_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant4matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  vecquant4matmul_sycl_old(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant8matmul_sycl_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant8matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  vecquant8matmul_sycl_old(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant2matmul_faster_sycl_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
); 

void vecquant2matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  vecquant2matmul_faster_sycl_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
}

void vecquant3matmul_faster_sycl_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
); 

void vecquant3matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  vecquant3matmul_faster_sycl_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
}

void vecquant4matmul_faster_sycl_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
); 

void vecquant4matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  vecquant4matmul_faster_sycl_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant2matmul", &vecquant2matmul, "Vector 2-bit Quantized Matrix Multiplication (sycl) (desc_act)");
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (sycl) (desc_act)");
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (sycl) (desc_act)");
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (sycl) (desc_act)");
  
  m.def("vecquant2matmul_old", &vecquant2matmul_old, "Vector 2-bit Quantized Matrix Multiplication (sycl)");
  m.def("vecquant3matmul_old", &vecquant3matmul_old, "Vector 3-bit Quantized Matrix Multiplication (sycl)");
  m.def("vecquant4matmul_old", &vecquant4matmul_old, "Vector 4-bit Quantized Matrix Multiplication (sycl)");
  m.def("vecquant8matmul_old", &vecquant8matmul_old, "Vector 8-bit Quantized Matrix Multiplication (sycl)");
  m.def("vecquant2matmul_faster_old", &vecquant2matmul_faster_old, "Vector 2-bit Quantized Matrix Multiplication (sycl), faster version");
  m.def("vecquant3matmul_faster_old", &vecquant3matmul_faster_old, "Vector 3-bit Quantized Matrix Multiplication (sycl), faster version");
  m.def("vecquant4matmul_faster_old", &vecquant4matmul_faster_old, "Vector 4-bit Quantized Matrix Multiplication (sycl), faster version");
}
