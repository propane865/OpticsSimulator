"""Vispy scene/node bookkeeping for the desktop application.

The :class:`SceneController` owns the render canvas, the view and every node
list (elements, rays, source marker, measurement planes) so the window and
editors can request redraws without juggling vispy node objects.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import rendering.visualize as visualize
from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, MeasurementResult
from optics.primitive import Ray

if TYPE_CHECKING:
    from optics.system import OpticalSystem

__all__ = ["SceneController", "make_display"]

RAY_LENGTH = visualize.DEFAULT_RAY_LENGTH


def make_display(
    elements: Sequence[RefractiveElement],
    rays: Sequence[Ray],
    ray_length: float = RAY_LENGTH,
    axis_length: float | None = None,
) -> Any:
    canvas = visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    canvas.unfreeze()
    canvas.view = view
    canvas.freeze()
    visualize.build_scene(
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
        self._plane_templates: list[visualize.MeshTemplate] = []
        self._selected_elements: tuple[int, ...] = ()
        self._selected_plane: int | None = None

    @property
    def canvas(self) -> Any:
        return self._canvas

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
            visualize.set_element_nodes_highlight(self._element_nodes[index], highlighted)

    def _tint_plane(self, index: int | None, highlighted: bool) -> None:
        if index is not None and 0 <= index < len(self._plane_nodes):
            visualize.set_measurement_nodes_highlight(self._plane_nodes[index], highlighted)

    # -- full / partial rebuilds --------------------------------------------
    def full_rebuild(
        self,
        elements: Sequence[RefractiveElement],
        rays: Sequence[Ray],
        ray_length: float = RAY_LENGTH,
    ) -> None:
        visualize.clear_scene(self._view)
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
        _, self._element_nodes, self._ray_nodes = visualize.build_scene(
            self._view, elements, rays, ray_length=ray_length
        )

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
        visualize.frame_view(
            self._view,
            visualize.scene_bounds(
                elements, source=source, planes=planes, ray_length=ray_length
            ),
        )

    def refresh_source(self, source: Any) -> None:
        visualize.remove_nodes(self._source_nodes)
        self._source_nodes = visualize.build_source_nodes(self._view, source)

    def retrace(
        self,
        system: OpticalSystem | Sequence[RefractiveElement],
        rays: Sequence[Ray],
        ray_length: float = RAY_LENGTH,
    ) -> None:
        """Redraw the rays through a persistent :class:`OpticalSystem`."""
        visualize.remove_nodes(self._ray_nodes)
        self._ray_nodes = visualize.build_ray_nodes(self._view, system, rays, ray_length=ray_length)

    def retrace_from_segments(
        self,
        starts: Any,
        ends: Any,
        counts: Any,
    ) -> None:
        """Redraw rays from a precomputed trace (shared with plane measurement)."""
        visualize.remove_nodes(self._ray_nodes)
        self._ray_nodes = visualize.build_ray_nodes_from_segments(self._view, starts, ends, counts)

    # -- elements ------------------------------------------------------------
    def add_element_nodes(self, element: RefractiveElement) -> None:
        self._element_nodes.append(visualize.build_element_nodes(self._view, element))

    def insert_element_nodes(self, index: int, element: RefractiveElement) -> None:
        """Attach a new element's visuals at ``index``, as a duplicate does.

        Every element from ``index`` up shifts one row with the stack, so the
        selection tint follows it (the newly inserted row is not yet selected).
        """
        self._element_nodes.insert(index, visualize.build_element_nodes(self._view, element))
        self._selected_elements = tuple(
            candidate + 1 if candidate >= index else candidate
            for candidate in self._selected_elements
        )

    def replace_element_nodes(self, index: int, element: RefractiveElement) -> None:
        visualize.remove_nodes(self._element_nodes[index])
        self._element_nodes[index] = visualize.build_element_nodes(self._view, element)
        # An applied edit swaps in untinted meshes: keep the edited element lit.
        if index in self._selected_elements:
            self._tint_element(index, True)

    def remove_element_nodes(self, index: int) -> None:
        visualize.remove_nodes(self._element_nodes.pop(index))
        # The removed element loses its tint, and every selected element behind
        # it shifts up with the stack.
        self._selected_elements = tuple(
            candidate - 1 if candidate > index else candidate
            for candidate in self._selected_elements
            if candidate != index
        )

    # -- measurement planes --------------------------------------------------
    def rebuild_planes(
        self,
        planes: Sequence[MeasurementPlane],
        results: Sequence[MeasurementResult],
    ) -> None:
        """Refresh every plane visual for fresh trace results.

        Planes whose geometry is unchanged are re-tinted in place (only the
        detector colors change on a trace), so a lens edit no longer rebuilds
        and re-uploads every detector mesh.
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
        key = visualize.measurement_geometry_key(plane)
        if index >= len(self._plane_nodes):
            nodes = visualize.build_measurement_nodes(self._view, plane, result)
            self._plane_nodes.append(nodes)
            self._plane_keys.append(key)
            self._plane_templates.append(visualize.measurement_mesh_template(nodes[0]))
        elif self._plane_keys[index] == key:
            visualize.update_measurement_colors(
                self._plane_nodes[index][0], result, self._plane_templates[index]
            )
        else:
            visualize.remove_nodes(self._plane_nodes[index])
            nodes = visualize.build_measurement_nodes(self._view, plane, result)
            self._plane_nodes[index] = nodes
            self._plane_keys[index] = key
            self._plane_templates[index] = visualize.measurement_mesh_template(nodes[0])
        if index == self._selected_plane:
            self._tint_plane(index, True)

    def _drop_plane_node(self, index: int) -> None:
        visualize.remove_nodes(self._plane_nodes.pop(index))
        self._plane_keys.pop(index)
        self._plane_templates.pop(index)
        selected = self._selected_plane
        if index == selected:
            self._selected_plane = None
        elif selected is not None and index < selected:
            self._selected_plane = selected - 1

    def add_plane_nodes(self, plane: MeasurementPlane, result: MeasurementResult) -> None:
        self._sync_plane_node(len(self._plane_nodes), plane, result)

    def replace_plane_nodes(
        self, index: int, plane: MeasurementPlane, result: MeasurementResult
    ) -> None:
        self._sync_plane_node(index, plane, result)

    def remove_plane_nodes(self, index: int) -> None:
        self._drop_plane_node(index)
