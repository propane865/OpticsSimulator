// The two value objects of the numpy boundary -- ``FanBuffers`` and
// ``PlaneBatch`` -- and their registration. Both validate once, at
// construction, and keep the arrays they borrow alive: a kernel that takes them
// takes one argument whose fields cannot be transposed, in place of a run of
// same-shaped positional arrays. The validation itself is the shared
// ``support.hpp`` readers, so no rule is stated twice.
#include <utility>

#include "register.hpp"
#include "support.hpp"

namespace optics::bindings {

FanBuffers::FanBuffers(py::array_t<double> origins, py::array_t<double> directions,
                       py::array_t<double> wavelengths, py::object powers)
    : origins_(std::move(origins)), dirs_(std::move(directions)), wavelengths_(std::move(wavelengths)) {
    const ConstDoubles od = read_doubles(origins_, 2, "origins");
    const ConstDoubles dd = read_doubles(dirs_, 2, "dirs");
    wl_ = read_doubles(wavelengths_, 1, "wavelengths");
    require(od.size % 3 == 0, "origins: expected shape (N, 3)");
    n_ = od.size / 3;
    require(dd.size == od.size, "dirs: expected the same shape as origins");
    require(wl_.size == n_, "wavelengths: expected shape (N,)");
    if (powers.is_none()) {
        return;  // a trace-only fan: asking it for powers is an error, not zeros
    }
    powers_ = py::cast<py::array_t<double>>(powers);
    pw_ = read_doubles(powers_, 1, "powers");
    has_powers_ = true;
    require(pw_.size == n_, "powers: expected shape (N,)");
}

PlaneBatch::PlaneBatch(py::array_t<double> po, py::array_t<double> pn, py::array_t<double> u_min,
                       py::array_t<double> u_max, py::array_t<double> v_min,
                       py::array_t<double> v_max, py::array_t<double> s_u, py::array_t<double> s_v,
                       py::array_t<double> t_u, py::array_t<double> t_v)
    : po_(std::move(po)), pn_(std::move(pn)), u_min_(std::move(u_min)), u_max_(std::move(u_max)),
      v_min_(std::move(v_min)), v_max_(std::move(v_max)), s_u_(std::move(s_u)), s_v_(std::move(s_v)),
      t_u_(std::move(t_u)), t_v_(std::move(t_v)) {
    planes_ = read_planes(po_, pn_, u_min_, u_max_, v_min_, v_max_, s_u_, s_v_, t_u_, t_v_);
}

void register_support(py::module_& m) {
    py::class_<FanBuffers>(m, "FanBuffers")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::object>(),
             py::arg("origins"), py::arg("directions"), py::arg("wavelengths"),
             py::arg("powers") = py::none(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             py::keep_alive<1, 4>())
        .def_property_readonly("count", &FanBuffers::count);
    py::class_<PlaneBatch>(m, "PlaneBatch")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::array_t<double>,
                      py::array_t<double>>(),
             py::arg("po"), py::arg("pn"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"),
             py::arg("v_max"), py::arg("s_u"), py::arg("s_v"), py::arg("t_u"), py::arg("t_v"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>(),
             py::keep_alive<1, 5>(), py::keep_alive<1, 6>(), py::keep_alive<1, 7>(),
             py::keep_alive<1, 8>(), py::keep_alive<1, 9>(), py::keep_alive<1, 10>(),
             py::keep_alive<1, 11>())
        .def_property_readonly("count", &PlaneBatch::count);
}

}  // namespace optics::bindings
