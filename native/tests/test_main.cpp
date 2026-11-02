// Native test entry point: runs every suite and reports the failure count.
#include <cstdio>

#include "check.hpp"

void test_expression();
void test_kernels();
void test_surface();
void test_analytic();
void test_element();
void test_stack();
void test_measurement();
void test_optimizer();
void test_optimizer_acceptance();

int main() {
    test_expression();
    test_kernels();
    test_surface();
    test_analytic();
    test_element();
    test_stack();
    test_measurement();
    test_optimizer();
    test_optimizer_acceptance();
    if (g_failures == 0) {
        std::printf("native tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d native test failures\n", g_failures);
    return 1;
}
