#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <random>


class mnist_loader {
private:
    std::vector<std::vector<float>> m_images;
    std::vector<int> m_labels;
    int m_size;
    int m_rows;
    int m_cols;
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng;

    std::uniform_int_distribution<> dist{1,10000};

    void load_images(std::string file, int num=0);
    void load_labels(std::string file, int num=0);
    int  to_int(char* p);

public:
    mnist_loader(std::string image_file, std::string label_file, int num);
    mnist_loader(std::string image_file, std::string label_file);
    ~mnist_loader();

    int size() { return m_size; }
    int rows() { return m_rows; }
    int cols() { return m_cols; }

    std::vector<float> images(int id) { return m_images[id]; }
    int labels(int id) { return m_labels[id]; }
    std::vector<float> image_segment();
};

#endif
