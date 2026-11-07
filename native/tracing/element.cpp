// The internal-bounce loop for one closed refractive element: entry
// refraction followed by refraction-or-internal-reflection at each face until
// the ray leaves the glass (see ``element_hit.cpp`` for the face assembly and
// ``element_build.cpp`` for the persistent element itself). The glass index is
// the Sellmeier index at the ray's wavelength, evaluated once per walk.

#include <cmath>
#include <cstddef>

#include "optics/common/constants.hpp"
#include "optics/tracing/direction.hpp"
#include "optics/tracing/trace.hpp"

namespace optics::native {

ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d,
                           double wavelength_nm) {
    return trace_element(elem, o, d, wavelength_nm, element_first_hit(elem, o, d, 0.0));
}

ElementTrace trace_element(const Element& elem, const Vec3& o, const Vec3& d, double wavelength_nm,
                           const ElementHit& entry) {
    const bool inside = element_contains_point(elem, o);
    ElementTrace out;
    if (!entry.found) {
        if (inside) {
            out.pts[0] = o;
            out.exit_o = o;
            out.exit_d = d;
            out.exited = true;
            out.n_pts = 1;
            out.has_incident = false;
            out.incident_pt = o;
            return out;
        }
        out.exit_o = o;
        out.exit_d = d;
        out.exited = false;
        out.n_pts = 0;
        out.has_incident = false;
        out.incident_pt = o;
        return out;
    }
    const double n = element_index(elem, wavelength_nm);
    if (std::isnan(n)) {
        // The material has no refractive index at this wavelength (a Sellmeier
        // pole, or a non-positive n^2): treat it as opaque. The ray stops at the
        // face it reached and reports no exit, exactly like a ray swallowed
        // inside the glass.
        out.exit_o = inside ? o : entry.p;
        out.exit_d = d;
        out.exited = false;
        out.n_pts = 1;
        out.pts[0] = inside ? o : entry.p;
        out.has_incident = !inside;
        out.incident_pt = entry.p;
        return out;
    }
    Vec3 cur_o;
    Vec3 cur_d;
    if (inside) {
        cur_o = o;
        cur_d = d;
        out.has_incident = false;
        out.incident_pt = o;
        out.pts[0] = o;
        out.n_pts = 1;
    } else {
        out.has_incident = true;
        out.incident_pt = entry.p;
        const auto refr = refract_or_reflect(d, orient(entry.n, d), 1.0, n);
        cur_o = entry.p + RAY_EPSILON * refr.first;
        cur_d = refr.first;
        out.pts[0] = entry.p;
        out.n_pts = 1;
    }
    for (int bounce = 0; bounce < MAX_INTERNAL_BOUNCES; ++bounce) {
        ElementHit face = element_first_hit(elem, cur_o, cur_d, MIN_HIT_T);
        if (!face.found) {
            out.exit_o = cur_o;
            out.exit_d = cur_d;
            out.exited = false;
            return out;
        }
        out.pts[static_cast<std::size_t>(out.n_pts)] = face.p;
        out.n_pts += 1;
        const auto refr = refract_or_reflect(cur_d, orient(face.n, cur_d), n, 1.0);
        if (refr.second) {
            out.exit_o = face.p;
            out.exit_d = refr.first;
            out.exited = true;
            return out;
        }
        cur_o = face.p + RAY_EPSILON * refr.first;
        cur_d = refr.first;
    }
    out.exit_o = cur_o;
    out.exit_d = cur_d;
    out.exited = false;
    return out;
}

}  // namespace optics::native
