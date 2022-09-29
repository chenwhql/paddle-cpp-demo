#include <paddle/extension.h>
#include <iostream>

void print_tensor(const paddle::Tensor& tensor) {
    std::cout << "Tensor(";
    std::cout << tensor.place() << ", "
              << tensor.dtype() << ")[";
    auto* tensor_data = tensor.data<float>();
    for (size_t i = 0; i < tensor.numel(); ++i) {
        if (i == tensor.numel()-1) {
            std::cout << tensor_data[i] << "]" << std::endl;
            break;
        }
        std::cout << tensor_data[i] << " ";
    }
}

int main (){
    std::cout << "Demo execute start" << std::endl;

    auto a = paddle::full({3, 4}, 2.0);
    auto b = paddle::full({4, 5}, 3.0);
    auto out = paddle::matmul(a, b);

    print_tensor(out);

    std::cout << "Demo execute start" << std::endl;

    return 0;
}