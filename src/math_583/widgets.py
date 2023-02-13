"""Widgets with continuous_update set to False."""

from functools import partial
from ipywidgets import widgets

__all__ = ["IntSlider", "FloatSlider"]

IntSlider = partial(widgets.IntSlider, continuous_update=False)
FloatSlider = partial(widgets.FloatSlider, continuous_update=False)
