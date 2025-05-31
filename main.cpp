#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <matplot/matplot.h>
#include <pybind11/complex.h>
using namespace matplot;
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.def("add", [](int i, int j) {
        return i + j;
    });

    m.def("sin", [](double f, double start, double end, int samples, double amp) {
        std::vector<double> t, y;
        double dt = (end - start) / samples;
        for (int i = 0; i < samples; ++i) {
            double ti = start + i * dt;
            t.push_back(ti);
            y.push_back(amp * std::sin(2 * pi * f * ti));
        }
        plot(t, y);
        grid(on);
        show();
        save("sin", "png");
        return y;
    }, py::arg("f"), py::arg("start"), py::arg("end"), py::arg("samples"), py::arg("amp"));

    m.def("cos", [](double f, double start, double end, int samples, double amp) {
        std::vector<double> t, y;
        double dt = (end - start) / samples;
        for (int i = 0; i < samples; ++i) {
            double ti = start + i * dt;
            t.push_back(ti);
            y.push_back(amp * std::cos(2 * pi * f * ti));
        }
        plot(t, y);
        grid(on);
        show();
        save("cos", "png");
        return y;
    }, py::arg("f"), py::arg("start"), py::arg("end"), py::arg("samples"), py::arg("amp"));

    m.def("pila", [](double f, double fs, int samples) {
        std::vector<double> n, faza;
        for (int i = 0; i < samples; ++i) {
            double ni = static_cast<double>(i);
            double phi = std::fmod(ni * 2.0 * pi * f / fs, 2.0 * pi);
            n.push_back(ni);
            faza.push_back(phi);
        }
        plot(n, faza);
        grid(on);
        show();
        save("piloksztaltny", "png");
    }, py::arg("f"), py::arg("fs"), py::arg("samples"));

    m.def("pro", [](double f, double start, double end, int samples, double amp) {
        std::vector<double> t, y;
        double dt = (end - start) / samples;
        for (int i = 0; i < samples; ++i) {
            double ti = start + i * dt;
            t.push_back(ti);
            y.push_back((std::cos(2 * pi * f * ti) < 0) ? -amp : amp);
        }
        plot(t, y);
        grid(on);
        show();
        save("prostokatny", "png");
        return y;
    }, py::arg("f"), py::arg("start"), py::arg("end"), py::arg("samples"), py::arg("amp"));

    m.def("dft", [](const std::vector<double>& signal, double start, double end, int samples) {
        std::vector<std::complex<double>> vdft;
        std::vector<double> t = linspace(start, end, samples);
        std::vector<double> yreal;
        for (int k = 0; k < samples; ++k) {
            std::complex<double> sum = 0;
            for (int n = 0; n < samples; ++n) {
                double angle = -2.0 * pi * k * n / samples;
                sum += signal[n] * std::exp(std::complex<double>(0, angle));
            }
            yreal.push_back(std::abs(sum));
            vdft.push_back(sum);
        }
        plot(t, yreal);
        grid(on);
        show();
        save("transformata", "png");
        return vdft;
    }, py::arg("signal"), py::arg("start"), py::arg("end"), py::arg("samples"));

    m.def("idft", [](const std::vector<std::complex<double>>& spectrum, int samples) {
        std::vector<double> time_signal, t = linspace(0, 1, samples);
        for (int n = 0; n < samples; ++n) {
            std::complex<double> sum = 0.0;
            for (int k = 0; k < samples; ++k) {
                double angle = 2.0 * pi * k * n / samples;
                sum += spectrum[k] * std::exp(std::complex<double>(0, angle));
            }
            time_signal.push_back((sum / static_cast<double>(samples)).real());
        }
        plot(time_signal);
        grid(on);
        show();
        save("odwrotna_transformata", "png");
        return time_signal;
    }, py::arg("spectrum"), py::arg("samples"));

    m.def("filter1d", [](const std::vector<double>& signal, int window_size) {
        size_t n = signal.size();
        if (window_size < 1 || window_size > static_cast<int>(n)) {
            throw std::runtime_error("Invalid window size.");
        }
        std::vector<double> filtered(n);
        int half = window_size / 2;
        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            int count = 0;
            for (int j = -half; j <= half; ++j) {
                int idx = static_cast<int>(i) + j;
                if (idx >= 0 && idx < static_cast<int>(n)) {
                    sum += signal[idx];
                    count++;
                }
            }
            filtered[i] = sum / count;
        }
        plot(filtered);
        grid(on);
        show();
        save("1D", "png");
        return filtered;
    }, py::arg("signal"), py::arg("window_size"));

    m.def("filtr2", []() {
        std::vector<std::vector<double>> input = {
            {1, 2, 3, 4, 5},
            {5, 6, 7, 8, 9},
            {9, 10, 11, 12, 13},
            {13, 14, 15, 16, 17},
            {17, 18, 19, 20, 21}
        };

        std::vector<std::vector<double>> kernel = {
            {0, -1, 0},
            {-1, 4, -1},
            {0, -1, 0}
        };

        int rows = input.size();
        int cols = input[0].size();
        int ksize = kernel.size();
        int offset = ksize / 2;

        std::vector<std::vector<double>> output(rows, std::vector<double>(cols, 0.0));

        for (int m = 0; m < rows; ++m) {
            for (int n = 0; n < cols; ++n) {
                double sum = 0.0;
                for (int i = 0; i < ksize; ++i) {
                    for (int j = 0; j < ksize; ++j) {
                        int x = m + i - offset;
                        int y = n + j - offset;
                        if (x >= 0 && x < rows && y >= 0 && y < cols) {
                            sum += input[x][y] * kernel[ksize - 1 - i][ksize - 1 - j];
                        }
                    }
                }
                output[m][n] = sum;
            }
        }

        std::vector<std::vector<double>> X(rows, std::vector<double>(cols));
        std::vector<std::vector<double>> Y(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                X[i][j] = j;
                Y[i][j] = i;
            }
        }

        surf(X, Y, output);
        xlabel("os x");
        ylabel("os y");
        zlabel("os z");
        colorbar();
        grid(on);
        show();
        save("2D", "png");
        return output;
    });
}
