"""Vispy scene/node bookkeeping for the desktop application.

The :class:`SceneController` owns the render canvas, the view and every node
list (elements, rays, source marker, measurement planes) so the window and
editors can request redraws without juggling vispy node objects -- including
the viewport's clear colour, the one piece of the theme vispy keeps to itself
(:meth:`SceneController.set_viewport_theme`).

The element and plane node lists are *derived* state: they mirror the model
lists, so callers never add or remove individual nodes. They hand over the
current model lists and :meth:`SceneController.sync_elements` /
:meth:`SceneController.sync_planes` work out what changed (by object identity
for elements, by geometry key for planes) and apply the insertions, moves,
replacements and removals themselves. That is the only way the two index-keyed
lists can be kept in step -- and the only way a caller cannot get the order of
a scene update and a state mutation wrong.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, MeasurementResult
from optics.primitive import Ray
from rendering.camera import frame_view, scene_bounds
from rendering.canvas import build_scene, clear_scene, create_canvas, remove_nodes
from rendering.measurements import (
    MeshTemplate,
    build_measurement_nodes,
    measurement_geometry_key,
    measurement_mesh_template,
    set_measurement_nodes_highlight,
    update_measurement_colors,
)
from rendering.rays import build_ray_nodes, build_ray_nodes_from_segments
from rendering.sources import build_source_nodes
from rendering.style import DEFAULT_RAY_LENGTH
from rendering.surfaces import build_element_nodes, set_element_nodes_highlight

if TYPE_CHECKING:
    from optics.system import OpticalSystem

__all__ = ["SceneController", "make_display", "sync_scene"]

RAY_LENGTH = DEFAULT_RAY_LENGTH


def make_display(
    elements: Sequence[RefractiveElement],
    rays: Sequence[Ray],
    ray_length: float = RAY_LENGTH,
    axis_length: float | None = None,
    bgcolor: Sequence[float] | None = None,
) -> Any:
    """The canvas and its view, cleared to ``bgcolor`` (RGBA floats, ``None`` = dark).

    ``bgcolor`` is the viewport's ground, which is a theme choice rather than a scene
    one: callers with a palette to read hand over :func:`ui.theme.viewport_bgcolor`, and
    the Qt-free ones take the dark default :func:`rendering.canvas.create_canvas` keeps.
    """
    canvas = create_canvas(bgcolor=bgcolor)
    view = canvas.central_widget.add_view(camera="turntable")
    canvas.unfreeze()
    canvas.view = view
    canvas.freeze()
    build_scene(
        view, elements, rays, ray_length=ray_length, axis_length=axis_length
    )
    return canvas


class SceneController:
    """Owns the vispy view and the node bookkeeping for every drawable kind."""

    def __init__(self, canvas: Any) -> None:
        self._canvas: Any = canvas
        self._view: Any = canvas.view
        self._source_nodes: list[Any] = []
        self._element_nodes: list[list[Any]] = []
        self._ray_nodes: list[Any] = []
        self._plane_nodes: list[list[Any]] = []
        self._plane_keys: list[tuple] = []
        self._plane_templates: list[MeshTemplate] = []
        # The model objects the element nodes were built from, index for index
        # with ``_element_nodes``: what :meth:`sync_elements` diffs against.
        self._element_model: list[RefractiveElement] = []
        self._selected_elements: tuple[int, ...] = ()
        self._selected_plane: int | None = None

    @property
    def canvas(self) -> Any:
        return self._canvas

    def set_viewport_theme(self, bgcolor: Sequence[float]) -> None:
        """Clear the viewport to ``bgcolor`` (RGBA floats) -- the theme's own ground.

        vispy holds the clear colour itself; no ``QPalette`` owns it and nothing
        repaints it for free, so the theme has to be handed over and the canvas asked to
        draw again. Only the ground changes: the nodes, their colors and the camera are
        untouched, so a light/dark flip costs one clear colour and one redraw.
        """
        self._canvas.bgcolor = tuple(bgcolor)
        self._canvas.update()

    # -- selection highlight -------------------------------------------------
    def select_elements(self, indices: Iterable[int] | None) -> None:
        """Tint the elements the editors are showing (empty clears the tint).

        A set of indices, so selecting a submodule lights every member while a
        single lens is just the one-element case.
        """
        selected = tuple(sorted({int(index) for index in indices or ()}))
        selected = tuple(index for index in selected if 0 <= index < len(self._element_nodes))
        if selected == self._selected_elements:
            return
        self._tint_elements(self._selected_elements, False)
        self._selected_elements = selected
        self._tint_elements(selected, True)

    def _tint_elements(self, indices: Iterable[int], highlighted: bool) -> None:
        for index in indices:
            self._tint_element(index, highlighted)

    def select_plane(self, index: int | None) -> None:
        """Rim-tint the selected measurement plane (``None`` clears it)."""
        if index is not None and not 0 <= index < len(self._plane_nodes):
            index = None
        if index == self._selected_plane:
            return
        self._tint_plane(self._selected_plane, False)
        self._selected_plane = index
        self._tint_plane(index, True)

    def _tint_element(self, index: int | None, highlighted: bool) -> None:
        if index is not None and 0 <= index < len(self._element_nodes):
            set_element_nodes_highlight(self._element_nodes[index], highlighted)

    def _tint_plane(self, index: int | None, highlighted: bool) -> None:
        if index is not None and 0 <= index < len(self._plane_nodes):
            set_measurement_nodes_highlight(self._plane_nodes[index], highlighted)

    # -- full / partial rebuilds --------------------------------------------
    def full_rebuild(
        self,
        elements: Sequence[RefractiveElement],
        rays: Sequence[Ray],
        ray_length: float = RAY_LENGTH,
    ) -> None:
        clear_scene(self._view)
        # ``clear_scene`` detaches the source marker and plane visuals too, but
        # only elements/rays are rebuilt here: drop their stale bookkeeping so a
        # later ``refresh_source``/``rebuild_planes`` re-attaches instead of
        # re-tinting a detached mesh (whose geometry key still matches).
        self._source_nodes = []
        self._plane_nodes = []
        self._plane_keys = []
        self._plane_templates = []
        # Freshly built nodes carry their own colors: forget a selection whose
        # nodes no longer exist.
        self._selected_elements = ()
        self._selected_plane = None
        _, self._element_nodes, self._ray_nodes = build_scene(
            self._view, elements, rays, ray_length=ray_length
        )
        self._element_model = list(elements)

    def frame_view(
        self,
        elements: Sequence[RefractiveElement],
        source: Any = None,
        planes: Sequence[MeasurementPlane] = (),
        ray_length: float = RAY_LENGTH,
    ) -> None:
        """Fit the camera to a freshly loaded scene.

        Called on project load/import only: the camera keeps the user's zoom
        and orbit across element edits and retraces.
        """
        frame_view(
            self._view,
            scene_bounds(
                elements, source=source, planes=planes, ray_length=ray_length
            ),
        )

    def refresh_source(self, source: Any) -> None:
        remove_nodes(self._source_nodes)
        self._source_nodes = build_source_nodes(self._view, source)

    def retrace(
        self,
        system: OpticalSystem | Sequence[RefractiveElement],
        rays: Sequence[Ray],
        ray_length: float = RAY_LENGTH,
    ) -> None:
        """Redraw the rays through a persistent :class:`OpticalSystem`."""
        remove_nodes(self._ray_nodes)
        self._ray_nodes = build_ray_nodes(self._view, system, rays, ray_length=ray_length)

    def retrace_from_segments(
        self,
        starts: Any,
        ends: Any,
        counts: Any,
        transmissions: Any = None,
    ) -> None:
        """Redraw rays from a precomputed trace (shared with plane measurement).

        ``transmissions`` dims each drawn segment by the power still travelling on
        it, so the picture and the detector readings come from one walk and cannot
        disagree about how much light a dark element let through."""
        remove_nodes(self._ray_nodes)
        self._ray_nodes = build_ray_nodes_from_segments(
            self._view, starts, ends, counts, transmissions
        )

    # -- elements ------------------------------------------------------------
    def sync_elements(
        self, elements: Sequence[RefractiveElement], *, geometry_changed: bool = True
    ) -> None:
        """Mirror ``elements`` in the scene, working out what changed.

        Comparison is by object identity, which is how the model edits: applying
        a lens builds a new :class:`RefractiveElement`, inserting a duplicate
        adds one, removing one drops it. So a row whose object differs is a
        rebuilt visual, an object missing from the list is a removal, and an
        object the scene has never seen is an insertion. ``geometry_changed`` is
        ``False`` for the edits whose meshes still show the same geometry (a
        material-only edit, or no edit at all): then only the recorded identities
        are adopted, so a lens tinted to a new index is not re-meshed and
        re-uploaded for nothing.
        """
        target = list(elements)
        previous = list(self._element_model)
        if not geometry_changed and len(previous) == len(target):
            self._element_model = target
            return

        kept = [element for element in previous if any(other is element for other in target)]
        fresh = [element for element in target if not any(other is element for other in previous)]
        if len(previous) == len(target):
            # Same number of rows: every differing row is a rebuilt element. A
            # reorder lands here too, and re-builds the two rows it swapped --
            # the shared mesher caches by geometry, so that is a tint refresh.
            for index, (before, now) in enumerate(zip(previous, target)):
                if before is not now:
                    self._replace_element_node(index, now)
        elif len(kept) == len(previous) and len(target) == len(kept) + len(fresh):
            # Pure growth (a duplicate, an import): insert each new element at
            # its own row. The old ones keep their relative order, so ascending
            # insertion keeps the node list aligned with the stack.
            for index, element in enumerate(target):
                if any(element is before for before in previous):
                    continue
                self._insert_element_node(index, element)
        elif not fresh and len(target) == len(kept):
            # Pure shrink: drop the rows whose element left the stack, highest
            # first so the indices still to come stay valid.
            for index in range(len(previous) - 1, -1, -1):
                if not any(element is previous[index] for element in target):
                    self._remove_element_node(index)
        else:
            # Anything else (a load, a cascade that both adds and replaces) is
            # not worth diffing: rebuild every element visual.
            self._rebuild_element_nodes(target)
        self._element_model = target

    def _rebuild_element_nodes(self, elements: Sequence[RefractiveElement]) -> None:
        remove_nodes(self._element_nodes)
        self._element_nodes = [build_element_nodes(self._view, element) for element in elements]
        self._selected_elements = ()

    def _insert_element_node(self, index: int, element: RefractiveElement) -> None:
        """Attach a new element's visuals at ``index``, as a duplicate does.

        Every element from ``index`` up shifts one row with the stack, so the
        selection tint follows it (the newly inserted row is not yet selected).
        """
        self._element_nodes.insert(index, build_element_nodes(self._view, element))
        self._selected_elements = tuple(
            candidate + 1 if candidate >= index else candidate
            for candidate in self._selected_elements
        )

    def _replace_element_node(self, index: int, element: RefractiveElement) -> None:
        remove_nodes(self._element_nodes[index])
        self._element_nodes[index] = build_element_nodes(self._view, element)
        # An applied edit swaps in untinted meshes: keep the edited element lit.
        if index in self._selected_elements:
            self._tint_element(index, True)

    def _remove_element_node(self, index: int) -> None:
        remove_nodes(self._element_nodes.pop(index))
        # The removed element loses its tint, and every selected element behind
        # it shifts up with the stack.
        self._selected_elements = tuple(
            candidate - 1 if candidate > index else candidate
            for candidate in self._selected_elements
            if candidate != index
        )

    # -- measurement planes --------------------------------------------------
    def sync_planes(
        self,
        planes: Sequence[MeasurementPlane],
        results: Sequence[MeasurementResult],
    ) -> None:
        """Mirror ``planes`` in the scene for fresh trace results.

        Planes whose geometry is unchanged are re-tinted in place (only the
        detector colors change on a trace), so a lens edit no longer rebuilds
        and re-uploads every detector mesh. Geometry is the key here rather than
        object identity: a plane re-applied with the same shape keeps its mesh,
        and a removal or insertion shifts the rows into place.
        """
        while len(self._plane_nodes) > len(planes):
            self._drop_plane_node(len(self._plane_nodes) - 1)
        for index, (plane, result) in enumerate(zip(planes, results)):
            self._sync_plane_node(index, plane, result)

    def _sync_plane_node(
        self, index: int, plane: MeasurementPlane, result: MeasurementResult
    ) -> None:
        """Attach/refresh one plane visual, keeping the selection rim lit.

        A geometry change rebuilds the nodes (untinted), so the highlight is
        re-asserted for whichever plane is selected; a color-only refresh leaves
        it in place already.
        """
        key = measurement_geometry_key(plane)
        if index >= len(self._plane_nodes):
            nodes = build_measurement_nodes(self._view, plane, result)
            self._plane_nodes.append(nodes)
            self._plane_keys.append(key)
            self._plane_templates.append(measurement_mesh_template(nodes[0]))
        elif self._plane_keys[index] == key:
            update_measurement_colors(
                self._plane_nodes[index][0], result, self._plane_templates[index]
            )
        else:
            remove_nodes(self._plane_nodes[index])
            nodes = build_measurement_nodes(self._view, plane, result)
            self._plane_nodes[index] = nodes
            self._plane_keys[index] = key
            self._plane_templates[index] = measurement_mesh_template(nodes[0])
        if index == self._selected_plane:
            self._tint_plane(index, True)

    def _drop_plane_node(self, index: int) -> None:
        remove_nodes(self._plane_nodes.pop(index))
        self._plane_keys.pop(index)
        self._plane_templates.pop(index)
        selected = self._selected_plane
        if index == selected:
            self._selected_plane = None
        elif selected is not None and index < selected:
            self._selected_plane = selected - 1

class SceneModel(Protocol):
    """The project lists a scene mirrors: the stack, the planes and their images."""

    @property
    def elements(self) -> list[RefractiveElement]: ...

    @property
    def planes(self) -> list[MeasurementPlane]: ...

    @property
    def plane_results(self) -> list[MeasurementResult]: ...


def sync_scene(model: SceneModel, scene: SceneController, *, geometry_changed: bool = True) -> None:
    """Bring the scene's element and plane visuals in line with ``model``.

    The single place both lists are synced, called by whatever mutated the
    model: the controllers never touch individual nodes, so there is no pair of
    index-keyed lists kept in step by hand and no ordering to get wrong between
    the state edit and the scene edit. ``geometry_changed`` comes from
    :func:`ui.project.elements.classify_element_edit` -- a material-only (or
    no-op) edit leaves every mesh showing the same geometry, so the element
    visuals are not rebuilt for it.
    """
    scene.sync_elements(model.elements, geometry_changed=geometry_changed)
    scene.sync_planes(model.planes, model.plane_results)
