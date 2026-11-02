"""Project-state components: the internal subsystems :class:`ui.model.ProjectState`
composes (measurement planes; element and persistence operations live here too).

These are not independent state objects: the aggregate root owns one instance
of each and delegates, so there is no cross-object synchronization to get
wrong.
"""
