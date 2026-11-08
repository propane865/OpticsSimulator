// Element-level kernel suite: nearest-face assembly (surfaces + sidewall),
// containment, refraction through a parallel-plane plate and total internal
// reflection.
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"
#include "optics/tracing/context.hpp"
#include "optics/tracing/trace.hpp"

#include "check.hpp"
#include "test_element.hpp"

namespace {

using optics::native::element_contains_point;
using optics::native::element_extinction;
using optics::native::element_first_hit;
using optics::native::element_index;
using optics::native::element_optical_response;
using optics::native::HC_EV_NM;
using optics::native::LorentzOscillator;
using optics::native::make_vec3;
using optics::native::material_complex_index;
using optics::native::material_dielectric;
using optics::native::material_optical_response;
using optics::native::norm;
using optics::native::PI;
using optics::native::Surface;
using optics::native::trace_element;
using optics::native::Vec3;

// sqrt(1 - 0.3^2): the incident ray below has sin(theta) = 0.3.
constexpr double kCos1 = 0.9539392014169457;

fixture::Element small_plate() {
    return fixture::make_plate(1.5, 1.0, -0.5, 0.5, 0.0, 48, true);
}

// The persistent ``Element::create`` path must derive the AABB, graph flags
// (owned by the surfaces) and material exactly as configured.
void test_create_persists_state() {
    auto face = [](double height, double half) {
        char z_src[64];
        std::snprintf(z_src, sizeof(z_src), "%g", height);
        Surface::Options o;
        o.x_src = "u";
        o.y_src = "v";
        o.z_src = z_src;
        o.u_min = -half;
        o.u_max = half;
        o.v_min = -half;
        o.v_max = half;
        o.grid_n = 48;
        return Surface::create(std::move(o));
    };
    const fixture::MaterialResponseRef material = fixture::constant_material(1.5);
    const auto elem = optics::native::Element::create(face(0.5, 1.0), face(-0.5, 1.0), material,
                                                     std::vector<double>{}, std::vector<double>{});
    CHECK(elem->surface1 != nullptr && elem->surface2 != nullptr);
    CHECK(elem->material == material);  // shared, never copied
    CHECK(close(element_index(*elem, fixture::kTestWavelengthNm), 1.5));
    // A wavelength-independent material holds its index at both ends of the band,
    // and absorbs nothing anywhere.
    CHECK(close(element_index(*elem, 300.0), 1.5));
    CHECK(close(element_index(*elem, 2000.0), 1.5));
    CHECK(close(element_extinction(*elem, 300.0), 0.0, 1e-15));
    CHECK(close(element_optical_response(*elem, 2000.0).alpha_per_mm, 0.0, 1e-15));
    CHECK(close(elem->lo[0], -1.0) && close(elem->lo[1], -1.0) && close(elem->lo[2], -0.5));
    CHECK(close(elem->hi[0], 1.0) && close(elem->hi[1], 1.0) && close(elem->hi[2], 0.5));
    CHECK(elem->surface1->is_graph_over_xy() && elem->surface2->is_graph_over_xy());
    CHECK(elem->n_tris() == 0);

    // Trace matches the fixture plate bit-for-bit.
    const optics::native::ElementHit hit =
        optics::native::element_first_hit(*elem, make_vec3(0.1, 0.2, 3.0),
                                          make_vec3(0.0, 0.0, -1.0), 0.0);
    CHECK(hit.found);
    CHECK(close(hit.t, 2.5) && close(hit.p[2], 0.5));

    // Inconsistent sidewall arrays are rejected.
    bool threw = false;
    try {
        optics::native::Element::create(face(0.5, 1.0), face(-0.5, 1.0), material,
                                       std::vector<double>(10, 0.0), std::vector<double>{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Materials are validated at construction too: a permittivity that is not
    // positive, or an oscillator with a nonsense resonance or damping, is refused
    // rather than traced with.
    const auto build = [&](const optics::native::MaterialResponseRef& m) {
        return optics::native::Element::create(face(0.5, 1.0), face(-0.5, 1.0), m,
                                              std::vector<double>{}, std::vector<double>{});
    };
    auto bad_epsilon = std::make_shared<optics::native::MaterialResponse>();
    bad_epsilon->epsilon_inf = 0.0;
    threw = false;
    try {
        build(bad_epsilon);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        build(fixture::oscillator_material(1.0, {LorentzOscillator{1.0, -2.0, 0.01}}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        build(fixture::oscillator_material(1.0, {LorentzOscillator{1.0, 2.0, -0.01}}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_first_hit() {
    const fixture::Element plate = small_plate();

    const Vec3 o = make_vec3(0.1, 0.2, 3.0);
    const Vec3 d = make_vec3(0.0, 0.0, -1.0);
    const optics::native::ElementHit hit = element_first_hit(plate, o, d, 0.0);
    CHECK(hit.found);
    CHECK(close(hit.t, 2.5));
    CHECK(close(hit.p[0], 0.1) && close(hit.p[1], 0.2) && close(hit.p[2], 0.5));
    // The normal convention is ``n = -cross(r_u, r_v)``, so a ``z = const``
    // face has world normal -z.
    CHECK(std::abs(hit.n[0]) < 1e-8 && std::abs(hit.n[1]) < 1e-8 && close(hit.n[2], -1.0));

    // Sidewall entry from the side (the plane faces are parallel to the ray).
    const Vec3 wo = make_vec3(-3.0, 0.0, 0.0);
    const Vec3 wd = make_vec3(1.0, 0.0, 0.0);
    const optics::native::ElementHit wall = element_first_hit(plate, wo, wd, 0.0);
    CHECK(wall.found);
    CHECK(close(wall.t, 2.0));
    CHECK(close(wall.p[0], -1.0) && close(wall.p[2], 0.0));
    CHECK(close(wall.n[0], -1.0) && std::abs(wall.n[1]) < 1e-12 && std::abs(wall.n[2]) < 1e-12);

    // ``min_t`` rejects the mesh hit as a whole: the kernel sees only the
    // nearest sidewall face, so a ``min_t`` past it hides the far wall too
    // (semantics the ``Element`` refactor must preserve).
    const optics::native::ElementHit far = element_first_hit(plate, wo, wd, 2.5);
    CHECK(!far.found);
}

void test_contains() {
    const fixture::Element plate = small_plate();
    CHECK(element_contains_point(plate, make_vec3(0.0, 0.1, 0.0)));
    CHECK(!element_contains_point(plate, make_vec3(0.0, 0.0, 0.6)));
    CHECK(!element_contains_point(plate, make_vec3(2.0, 0.0, 0.0)));
    CHECK(!element_contains_point(plate, make_vec3(0.0, 0.0, 0.5)));
}

void test_trace_straight() {
    const fixture::Element plate = small_plate();
    const Vec3 o = make_vec3(0.0, 0.0, 3.0);
    const Vec3 d = make_vec3(0.0, 0.0, -1.0);
    const optics::native::ElementTrace tr =
        trace_element(plate, o, d, fixture::kTestWavelengthNm);
    CHECK(tr.exited);
    CHECK(tr.has_incident);
    CHECK(tr.n_pts == 2);
    CHECK(close(tr.incident_pt[2], 0.5));
    CHECK(close(tr.exit_o[2], -0.5));
    CHECK(close(tr.exit_o[0], 0.0, 1e-12) && close(tr.exit_o[1], 0.0, 1e-12));
    CHECK(close(tr.exit_d[2], -1.0, 1e-12) && std::abs(tr.exit_d[0]) < 1e-12);
}

void test_trace_oblique() {
    const fixture::Element plate = small_plate();
    const Vec3 o = make_vec3(-0.2, 0.0, 2.0);
    const Vec3 d = make_vec3(0.3, 0.0, -kCos1);
    const optics::native::ElementTrace tr =
        trace_element(plate, o, d, fixture::kTestWavelengthNm);
    CHECK(tr.exited);
    CHECK(tr.n_pts == 2);
    // Snell (1.0 -> 1.5): entry point from the free path, lateral drift from
    // the internal slope ``sin/cos = 0.2 / sqrt(1 - 0.2^2)`` over ``dz = 1``.
    const double t_entry = 1.5 / kCos1;
    const double x_entry = -0.2 + 0.3 * t_entry;
    const double tan_in = 0.2 / std::sqrt(1.0 - 0.04);
    const double x_exit = x_entry + tan_in * 1.0;
    CHECK(close(tr.incident_pt[0], x_entry, 1e-8));
    CHECK(close(tr.incident_pt[2], 0.5, 1e-8));
    CHECK(close(tr.exit_o[0], x_exit, 1e-8));
    CHECK(close(tr.exit_o[2], -0.5, 1e-8));
    // A parallel-plane plate deflects the ray back onto the incident
    // direction (lateral shift only).
    CHECK(close(tr.exit_d[0], 0.3, 1e-9));
    CHECK(close(tr.exit_d[2], -kCos1, 1e-9));
    CHECK(close(norm(tr.exit_d), 1.0, 1e-12));
}

void test_trace_tir() {
    // Tall plate (no sidewalls): a 45-degree internal ray exceeds the
    // critical angle ``asin(1/1.5) ~ 41.8`` and is trapped at every face.
    const fixture::Element slab =
        fixture::make_plate(1.5, 12.0, -0.5, 0.5, 0.0, 256, false);
    const Vec3 o = make_vec3(0.0, 0.0, 0.0);
    const double s = 1.0 / std::sqrt(2.0);
    const Vec3 d = make_vec3(s, 0.0, s);  // > critical angle from the normal
    CHECK(element_contains_point(slab, o));

    const optics::native::ElementTrace tr =
        trace_element(slab, o, d, fixture::kTestWavelengthNm);
    CHECK(!tr.exited);  // total internal reflection at every bounce
    CHECK(!tr.has_incident);  // the ray started inside the glass
    CHECK(tr.n_pts == optics::native::MAX_INTERNAL_BOUNCES + 1);
    for (int k = 1; k < tr.n_pts; ++k) {
        CHECK(close(std::abs(tr.pts[static_cast<std::size_t>(k)][2]), 0.5, 1e-8));
        // 45-degree zig-zag: hits at x = 0.5, 1.5, 2.5, ...
        CHECK(close(tr.pts[static_cast<std::size_t>(k)][0], k - 0.5, 1e-7));
        CHECK(close(tr.pts[static_cast<std::size_t>(k)][1], 0.0, 1e-12));
    }
}

// A transform-only element variant must trace bit-identically to a from-scratch
// rebuild at the same world placement (translated plate vs. rebuilt plate).
void test_variant_trace_equivalence() {
    const auto base = fixture::make_plate_ptr(1.5, 1.0, -0.5, 0.5, 2.0, 48, true);
    const auto moved = optics::native::Element::translated(*base, make_vec3(0.0, 0.0, -5.0));
    const auto scratch =
        fixture::make_plate_ptr(1.5, 1.0, -0.5, 0.5, -3.0, 48, true);

    CHECK(moved->box_valid == scratch->box_valid);
    CHECK(moved->wall_valid == scratch->wall_valid);
    for (int j = 0; j < 3; ++j) {
        CHECK(close(moved->box_lo[j], scratch->box_lo[j], 1e-12));
        CHECK(close(moved->box_hi[j], scratch->box_hi[j], 1e-12));
    }
    CHECK(moved->n_tris() == scratch->n_tris());

    const Vec3 o = make_vec3(0.1, -0.05, 6.0);
    const Vec3 d = make_vec3(0.05, 0.02, -1.0);
    const auto hv = element_first_hit(*moved, o, d, 0.0);
    const auto hs = element_first_hit(*scratch, o, d, 0.0);
    CHECK(hv.found && hs.found);
    CHECK(close(hv.t, hs.t, 1e-10));
    for (int j = 0; j < 3; ++j) {
        CHECK(close(hv.p[j], hs.p[j], 1e-10));
        CHECK(close(hv.n[j], hs.n[j], 1e-10));
    }
    const auto mv = trace_element(*moved, o, d, fixture::kTestWavelengthNm);
    const auto ms = trace_element(*scratch, o, d, fixture::kTestWavelengthNm);
    CHECK(mv.exited == ms.exited);
    CHECK(mv.n_pts == ms.n_pts);
    for (int j = 0; j < 3; ++j) {
        CHECK(close(mv.exit_o[j], ms.exit_o[j], 1e-10));
        CHECK(close(mv.exit_d[j], ms.exit_d[j], 1e-12));
    }
}

// The Lorentz dielectric model: epsilon from the oscillator sum, the passive
// square root, the Beer-Lambert coefficient, and the wavelengths with no answer.
void test_dielectric_response() {
    // One damped oscillator, evaluated here as separated real and imaginary parts
    // (1 / (a - i b) = (a + i b) / (a^2 + b^2)) so the check only agrees with the
    // kernel if the damping term carries the passive sign.
    const auto single = fixture::oscillator_material(2.0, {LorentzOscillator{1.2, 2.0, 0.3}});
    const double wavelength = 700.0;
    const double energy = HC_EV_NM / wavelength;
    const double re = 4.0 - energy * energy;
    const double im = 0.3 * energy;
    const double square = re * re + im * im;
    const std::complex<double> expected(2.0 + 1.2 * 4.0 * re / square, 1.2 * 4.0 * im / square);
    const std::complex<double> epsilon = material_dielectric(*single, wavelength);
    CHECK(close(epsilon.real(), expected.real(), 1e-13));
    CHECK(close(epsilon.imag(), expected.imag(), 1e-13));

    // The optical response is the passive square root of that same epsilon, taken
    // once: (n + ik)^2 = epsilon with n >= 0 and k >= 0, and alpha = 4 pi k /
    // lambda in inverse millimetres.
    const optics::native::OpticalResponse response = material_optical_response(*single, wavelength);
    const std::complex<double> root = material_complex_index(*single, wavelength);
    CHECK(close(response.n, root.real(), 1e-15) && close(response.k, root.imag(), 1e-15));
    CHECK(response.n >= 0.0 && response.k >= 0.0);
    const std::complex<double> squared = root * root;
    CHECK(close(squared.real(), epsilon.real(), 1e-12));
    CHECK(close(squared.imag(), epsilon.imag(), 1e-12));
    CHECK(close(response.alpha_per_mm, 4.0 * PI * response.k / (wavelength * 1.0e-6), 1e-12));

    // A dispersive crown: three oscillators, index falling with wavelength and
    // extinction that stays tiny across the visible band.
    const auto crown = fixture::oscillator_material(
        1.64143538, {LorentzOscillator{0.42758017, 10.2, 1.0e-5},
                     LorentzOscillator{0.2, 8.2, 1.0e-5},
                     LorentzOscillator{0.6, 0.11, 3.0e-4}});
    const double n_blue = material_complex_index(*crown, 486.13).real();
    const double n_green = material_complex_index(*crown, 587.56).real();
    const double n_red = material_complex_index(*crown, 656.27).real();
    CHECK(n_blue > n_green && n_green > n_red);
    CHECK(close(n_green, 1.5168, 2e-4));
    const double k_green = material_complex_index(*crown, 550.0).imag();
    CHECK(k_green > 0.0);
    CHECK(k_green < 1.0e-5);

    // An undamped resonance probed exactly on top of it has no response at all --
    // NaN, which the trace turns into an absorbed ray rather than a nonsense index.
    const auto undamped = fixture::oscillator_material(1.0, {LorentzOscillator{1.0, 2.0, 0.0}});
    const double resonant_wavelength = HC_EV_NM / 2.0;
    CHECK(std::isnan(material_complex_index(*undamped, resonant_wavelength).real()));
    CHECK(std::isnan(material_optical_response(*undamped, resonant_wavelength).n));
    const std::complex<double> off_resonance =
        material_complex_index(*undamped, resonant_wavelength * 1.3);
    CHECK(std::isfinite(off_resonance.real()));
    CHECK(close(off_resonance.imag(), 0.0, 1e-15));

    // Probing an undamped oscillator from above flips the permittivity negative:
    // the passive root is then purely imaginary, n = 0 with k > 0. The trace treats
    // such a material as opaque; the model states it instead of hiding it.
    const auto strong = fixture::oscillator_material(1.0, {LorentzOscillator{3.0, 1.0, 0.0}});
    const std::complex<double> metal = material_complex_index(*strong, HC_EV_NM / 1.5);
    CHECK(close(metal.real(), 0.0, 1e-12));
    CHECK(metal.imag() > 0.0);

    // A wavelength the model cannot interpret is reported the same way.
    CHECK(std::isnan(material_complex_index(*crown, 0.0).real()));
    CHECK(std::isnan(material_complex_index(*crown, -500.0).real()));
    CHECK(std::isnan(material_complex_index(*crown, std::numeric_limits<double>::quiet_NaN()).real()));
}

// A dispersive element: the parallel-plane plate leaves the exit direction
// untouched but shifts each colour by its own amount, and swallows the ray whose
// wavelength lands on an undamped resonance.
void test_dispersive_element_trace() {
    const auto crown = fixture::oscillator_material(
        1.64143538, {LorentzOscillator{0.42758017, 10.2, 1.0e-5},
                     LorentzOscillator{0.2, 8.2, 1.0e-5},
                     LorentzOscillator{0.6, 0.11, 3.0e-4}});
    const auto plate =
        optics::native::Element::create(fixture::plate_face(0.5, 1.0, 0.0, 48),
                                       fixture::plate_face(-0.5, 1.0, 0.0, 48), crown,
                                       std::vector<double>{}, std::vector<double>{});
    const Vec3 o = make_vec3(-0.2, 0.0, 2.0);
    const Vec3 d = make_vec3(0.3, 0.0, -kCos1);
    const auto blue = trace_element(*plate, o, d, 486.13);
    const auto red = trace_element(*plate, o, d, 656.27);
    CHECK(blue.exited && red.exited);
    CHECK(close(blue.exit_d[0], red.exit_d[0], 1e-12));
    CHECK(close(norm(blue.exit_d), 1.0, 1e-12));
    // The stronger index bends the internal ray closer to the normal, so blue
    // drifts less across the plate than red.
    CHECK(blue.exit_o[0] < red.exit_o[0]);

    // A plate whose material resonates without damping at 700 nm stops that ray at
    // the face it reached and traces normally at any other wavelength.
    const auto resonant = optics::native::Element::create(
        fixture::plate_face(0.5, 1.0, 0.0, 48), fixture::plate_face(-0.5, 1.0, 0.0, 48),
        fixture::oscillator_material(1.0, {LorentzOscillator{1.0, HC_EV_NM / 700.0, 0.0}}),
        std::vector<double>{}, std::vector<double>{});
    const Vec3 straight = make_vec3(0.0, 0.0, 3.0);
    const Vec3 down = make_vec3(0.0, 0.0, -1.0);
    const auto trapped = trace_element(*resonant, straight, down, 700.0);
    CHECK(!trapped.exited);
    CHECK(trapped.has_incident);
    CHECK(close(trapped.exit_o[2], 0.5, 1e-12));
    CHECK(close(trapped.exit_d[2], -1.0, 1e-12));
    const auto passed = trace_element(*resonant, straight, down, 900.0);
    CHECK(passed.exited);
    CHECK(close(passed.exit_o[2], -0.5, 1e-9));
}

}  // namespace

void test_element() {
    test_create_persists_state();
    test_first_hit();
    test_contains();
    test_trace_straight();
    test_trace_oblique();
    test_trace_tir();
    test_variant_trace_equivalence();
    test_dielectric_response();
    test_dispersive_element_trace();
}
