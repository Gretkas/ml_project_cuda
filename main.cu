#include "helper.cu"
#include "ojas.cu"
#include <iostream>
#include <vector>
#include <cuda_profiler_api.h>

int main() {
    const int num_neurons = 100;                    //ant nevroner som trenes
    const int num_seg = 100;                        //ant segmenter/bilder som algoritmen skal trenes med
    const int len = (5*5); //evt (28*28)            //lengden på et patch/bilde: Patch:(5*5), Bilde:(28*28)
    float *w = generate_w(len * num_neurons);       //skal bare være  (lengde på segment) * (ant nevroner)

    std::cout << "w(0):" << std::endl;
    std::cout << w << std::endl
              << std::endl;

    std::vector<float> x = load_data(num_seg, true); //Set lik false dersom du ønsker å bruke bilder istedenfor bildepatcher/segmenter

    cudaProfilerStart();
    run_ojas(w, x, num_seg, len, false, num_neurons); //Bør settes til true dersom man ønsker å regne ut y på en parallellisert måte
    cudaProfilerStop();

    std::cout << "w(" << num_seg << "):" << std::endl;
    std::cout << w << std::endl;
}
