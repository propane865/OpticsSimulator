// Persistent trace-stack state: a refractive element assembled from native
// surfaces (built once at element construction) and the stack of elements the
// trace kernels walk read-only.
#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "optics/common/constants.hpp"
#include "optics/common/vector.hpp"
#include "optics/geometry/surface.hpp"

namespace optics::native {

// One Lorentz resonance of a material: dimensionless ``strength``, resonance and
// damping in electronvolts. Storing energies (not angular frequencies) keeps the
// numbers a project file states in the range a reader recognises, and turns a
// ray's nanometre wavelength into the model's probe energy with one division by
// ``HC_EV_NM``.
struct LorentzOscillator {
    double strength = 0.0;
    double resonance_ev = 0.0;
    double damping_ev = 0.0;
};

// A material's complex dielectric response: the background permittivity plus the
// oscillator sum. Held by shared pointer because an element is copied constantly
// (the layout optimizer's sweep translates one element per candidate) while its
// material never changes; sharing the response keeps that copy allocation-free.
struct MaterialResponse {
    double epsilon_inf = 1.0;
    std::vector<LorentzOscillator> oscillators;
};

using MaterialResponseRef = std::shared_ptr<const MaterialResponse>;

// The material a default-constructed element carries: vacuum. A background
// permittivity of one with no oscillators is exactly ``n = 1``, so an element that
// was never given a material bends nothing instead of reading garbage.
inline MaterialResponseRef default_material() {
    static const MaterialResponseRef vacuum = std::make_shared<const MaterialResponse>();
    return vacuum;
}

// Non-finite sentinel for "this material has no optical response at this
// wavelength" (a non-positive wavelength, or an undamped resonance landed on).
// Callers absorb the ray rather than propagate NaN geometry, exactly as the
// Python model raises instead of returning an index.
inline std::complex<double> no_response() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan};
}

// One closed refractive element: two persistent surfaces (each owning its
// descriptor, seed grid and graph-over-xy flag, which the kernels read
// through the surfaces), the world-space sidewall mesh and the element-local
// axis-aligned bounds sampled once at construction.
class Element {
  public:
    // Sidewall triangles ``(K, 3, 3)`` and unit normals ``(K, 3)`` are
    // world-space, matching the surfaces' transforms; empty for a wall-less
    // element. Throws ``std::invalid_argument`` on inconsistent inputs.
    static std::shared_ptr<const Element> create(std::shared_ptr<const Surface> surface1,
                                                 std::shared_ptr<const Surface> surface2,
                                                 MaterialResponseRef material,
                                                 std::vector<double> tri,
                                                 std::vector<double> normals);

    // Pure-translation variant of an existing element: surfaces via the
    // transform-only fast path, world-space wall triangles shifted by
    // ``delta``, and the world box and wall band rebuilt from the shifted
    // data. Equivalent to an element built with both transforms translated
    // by ``delta``; the layout-optimizer sweep path.
    static std::shared_ptr<const Element> translated(const Element& base, const Vec3& delta);

    std::shared_ptr<const Surface> surface1;
    std::shared_ptr<const Surface> surface2;
    // Material: the Lorentz dielectric response, evaluated per ray at that
    // ray's wavelength (see ``element_optical_response``). Never null -- an
    // element without a material is vacuum (see ``default_material``).
    MaterialResponseRef material = default_material();
    Vec3 lo{};  // element-local AABB (from the surface models, built once)
    Vec3 hi{};
    std::vector<double> tri;      // (K, 3, 3) flattened row-major
    std::vector<double> normals;  // (K, 3) flattened row-major

    // Conservative world AABB over every traceable surface/wall point (seed
    // grid samples + wall vertices + a margin covering inter-sample
    // excursions). ``box_valid`` false disables the reject test entirely.
    bool box_valid = false;
    Vec3 box_lo{};
    Vec3 box_hi{};
    // Sidewall band in the surface1-local frame: the wall lives at radial
    // distance >= wall_r_min with local z inside [wall_z_lo, wall_z_hi].
    // ``wall_valid`` false (no wall or degenerate band) keeps the mesh test
    // unconditional.
    bool wall_valid = false;
    double wall_r_min = 0.0;
    double wall_z_lo = 0.0;
    double wall_z_hi = 0.0;

    std::size_t n_tris() const { return tri.size() / 9; }

    Element() = default;
};

using ElementRef = std::shared_ptr<const Element>;

// --- Material response -------------------------------------------------------
//
// The dielectric function belongs to the material, not to the element carrying
// it, so the evaluators take a ``MaterialResponse`` and the ``element_*`` wrappers
// at the end of this block only dereference an element's shared material. One
// definition of the physics then serves the trace, the native tests and (through
// the bindings and ``optics.primitive.Material``) the Python side.

// Everything one ray/element interaction needs from a material, from a single
// evaluation of the dielectric function: the index that refracts, the extinction
// coefficient, and the Beer-Lambert coefficient that turns it into power loss.
//
// ``alpha_per_mm`` is ``4 pi k / lambda`` with the wavelength converted to
// millimetres, because geometry in this simulator is millimetres -- the same
// quantity ``optics.primitive.Material.absorption_coefficient_at`` reports.
struct OpticalResponse {
    double n = 1.0;
    double k = 0.0;
    double alpha_per_mm = 0.0;
};

// Complex permittivity at ``wavelength_nm`` (nanometres):
//
//     epsilon(E) = epsilon_inf + sum_j S_j E_j^2 / (E_j^2 - E^2 - i Gamma_j E)
//
// with ``E = HC_EV_NM / wavelength_nm``. The damping term carries the minus sign,
// which is what makes ``Im[epsilon]`` positive for a passive medium and therefore
// ``k`` positive. NaN on a wavelength the model cannot answer.
inline std::complex<double> material_dielectric(const MaterialResponse& material,
                                                double wavelength_nm) {
    if (!std::isfinite(wavelength_nm) || wavelength_nm <= 0.0) {
        return no_response();
    }
    const double energy = HC_EV_NM / wavelength_nm;
    std::complex<double> epsilon(material.epsilon_inf, 0.0);
    for (const auto& osc : material.oscillators) {
        const double e0 = osc.resonance_ev;
        const std::complex<double> denominator(e0 * e0 - energy * energy,
                                              -osc.damping_ev * energy);
        if (denominator == 0.0) {
            return no_response();  // an undamped resonance exactly here
        }
        epsilon += (osc.strength * e0 * e0) / denominator;
    }
    if (!std::isfinite(epsilon.real()) || !std::isfinite(epsilon.imag())) {
        return no_response();
    }
    return epsilon;
}

// ``n + i k`` at ``wavelength_nm``, on the passive branch. A complex square root
// has two signs and the host's choice between them is a convention, so both parts
// are normalized here: a passive material returns ``n >= 0`` and ``k >= 0``
// regardless of the branch the host picked.
inline std::complex<double> material_complex_index(const MaterialResponse& material,
                                                   double wavelength_nm) {
    std::complex<double> root = std::sqrt(material_dielectric(material, wavelength_nm));
    if (!std::isfinite(root.real()) || !std::isfinite(root.imag())) {
        return no_response();
    }
    if (root.real() < 0.0) {
        root = -root;
    }
    if (root.imag() < 0.0) {
        root = std::complex<double>(root.real(), -root.imag());
    }
    return root;
}

// The full response of ``material`` at ``wavelength_nm``: epsilon and its square
// root are each evaluated once, and ``n``, ``k`` and ``alpha`` all come from that.
inline OpticalResponse material_optical_response(const MaterialResponse& material,
                                                 double wavelength_nm) {
    const std::complex<double> root = material_complex_index(material, wavelength_nm);
    OpticalResponse response;
    response.n = root.real();
    response.k = root.imag();
    response.alpha_per_mm = 4.0 * PI * root.imag() / (wavelength_nm * 1.0e-6);
    return response;
}

// Per-element conveniences over the three evaluators above.

inline std::complex<double> element_dielectric(const Element& elem, double wavelength_nm) {
    return material_dielectric(*elem.material, wavelength_nm);
}

inline std::complex<double> element_complex_index(const Element& elem, double wavelength_nm) {
    return material_complex_index(*elem.material, wavelength_nm);
}

// Refractive index of ``elem``'s material at ``wavelength_nm``. NaN when the
// material has no response there (see ``material_dielectric``).
inline double element_index(const Element& elem, double wavelength_nm) {
    return material_complex_index(*elem.material, wavelength_nm).real();
}

// Extinction coefficient of ``elem``'s material at ``wavelength_nm``.
inline double element_extinction(const Element& elem, double wavelength_nm) {
    return material_complex_index(*elem.material, wavelength_nm).imag();
}

inline OpticalResponse element_optical_response(const Element& elem, double wavelength_nm) {
    return material_optical_response(*elem.material, wavelength_nm);
}

// Rectangular detector extent in world ``(x, y)`` plane parameters (the
// ``u``/``v`` ranges a measurement plane accepts hits within).
struct PlaneRegion {
    double u_min = 0.0;
    double u_max = 0.0;
    double v_min = 0.0;
    double v_max = 0.0;
};

// A stack of elements traced as a unit. The historical tuple layout padded
// this to fixed capacities; a native vector carries the real element count
// directly.
struct TraceStack {
    std::vector<ElementRef> elements;

    std::size_t size() const { return elements.size(); }
};

}  // namespace optics::native
