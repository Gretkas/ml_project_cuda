#include "MNIST/mnist_loader.cpp"
#include <random>
#include <vector>

// From: https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
template <typename T>
T random(T range_from, T range_to) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

//for å printe ut arrays
std::ostream &operator<<(std::ostream &out, float *array) {
    const int len = 25; // må forandres etter hva x er

    for (int i = 0; i < len; i++) {
        out << array[i] << ", ";
    }
    return out;
}

std::vector<float> load_data(const int num, const bool return_segment = true, int img_quantity = 100) {
    if (!return_segment && num > img_quantity)
        img_quantity = num;

    mnist_loader train("datasets/train-images.idx3-ubyte",
                       "datasets/train-labels.idx1-ubyte", img_quantity);
    mnist_loader test("datasets/t10k-images.idx3-ubyte",
                      "datasets/t10k-labels.idx1-ubyte", img_quantity);

    std::vector<float> img;

    if (return_segment) {
        for (int i = 0; i < num; ++i) {
            std::vector<float> segment = train.image_segment();
            img.insert(img.end(), segment.begin(), segment.end());
        }
    } else {
        for (int i = 0; i < num; ++i) {
            int index = random(0, img_quantity);
            std::vector<float> segment = train.images(index);
            img.insert(img.end(), segment.begin(), segment.end());
        }
    }

    return img;
}

float *generate_w(const int len) {
    float *w;
    w = new float[len];
    srand((unsigned)time(0));

    for (int i = 0; i < len; ++i) {
        w[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (1 + 1)) - 1.0); //initailiserer mellom  -1 - 1
    }
    return w;
}
