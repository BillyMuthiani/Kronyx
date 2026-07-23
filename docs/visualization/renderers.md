# Renderers

Renderers consume a `StyledScene` and produce output. They are pure painters with no styling logic, no theme logic, and no defaults.

## Native SVG Renderer

The `SvgRenderer` is the default renderer. It draws rectangles, text, edges, and icons based solely on the properties present in a `StyledScene`.

```python
from kronyx.viz.renderers.svg import SvgRenderer
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.layout.vertical import VerticalLayout
from kronyx.viz.graph import Graph, Node, Edge

graph = Graph(
    nodes=[
        Node(id="input", name="Input", layer_type="Input", params=0, output_shape="?"),
        Node(id="dense", name="Dense", layer_type="Dense", params=48, output_shape="(None, 16)", metadata={"units": 16}),
        Node(id="output", name="Prediction", layer_type="Output", params=0, output_shape="?"),
    ],
    edges=[
        Edge(source="input", target="dense"),
        Edge(source="dense", target="output"),
    ],
)

scene = VerticalLayout().layout(graph)
styled_scene = DefaultSceneStyler().apply(scene)
renderer = SvgRenderer()
svg = renderer.render(styled_scene)
```

## Renderer Registry

Renderers are registered in a `RendererRegistry` and looked up by name at runtime.

```python
from kronyx.viz.registry import RendererRegistry
from kronyx.viz.renderers.svg import SvgRenderer

registry = RendererRegistry()
registry.register("svg", SvgRenderer)

renderer_cls = registry.get("svg")
renderer = renderer_cls()
```

## Future Renderers

The renderer abstraction is designed to support additional output formats without changing the pipeline:

- **PNG** — raster export via an image library
- **PDF** — vector document export
- **HTML** — interactive web-based diagrams
- **ASCII** — terminal-friendly text diagrams

All future renderers will consume the same `StyledScene` produced by the existing `SceneStyler` and `Theme` system.
