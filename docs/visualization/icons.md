# Icons

Icons provide semantic identity to layer types. They are renderer-independent and theme-independent.

## Built-in Icons

| Layer Type | Icon ID | Description |
|:-----------|:--------|:------------|
| Input | `input` | Open circle |
| Output | `output` | Filled circle |
| Dense | `dense` | Rectangle |
| Conv2D | `conv2d` | Grid |
| Conv1D | `conv1d` | Rectangle with horizontal line |
| Flatten | `flatten` | Rectangle with right arrow |
| Dropout | `dropout` | Rectangle with down arrow |
| BatchNormalization | `batch_norm` | Filled circle with inner dot |
| ReLU, Sigmoid, Softmax, Tanh, LeakyReLU | `relu`, `sigmoid`, `softmax`, `tanh`, `leaky_relu` | Lightning bolt |
| MaxPooling2D | `max_pool` | Stacked rectangles |
| AveragePooling2D | `avg_pool` | Stacked rectangles with dividing line |
| UnknownLayer | `unknown` | Diamond (fallback) |

## How Icons Work

Icons are semantic identifiers attached to `StyledNode` objects by the `SceneStyler`. Renderers receive pre-resolved icons and never perform layer-type lookups.

```
Layer Type (e.g. "Dense")
    ↓ IconRegistry.get("Dense")
Icon (id="dense", category="layer")
    ↓ Styler attaches to StyledNode
StyledNode(icon="dense")
    ↓ Renderer consumes
SVG <rect> or PNG bitmap or HTML <svg>
```

## Usage

Icons are automatically attached to nodes by the default styler. No manual configuration is required:

```python
from kronyx import Sequential, Dense, ReLU
from kronyx.viz.engine import VisualizationEngine

model = Sequential()
model.add(Dense(2, 16))
model.add(ReLU())

engine = VisualizationEngine()
engine.visualize(model, renderer_name="svg")
```

Each node in the output SVG will contain the icon corresponding to its layer type.

## Custom Icons

Register custom icons for new layer types:

```python
from kronyx.viz.icons import Icon, IconRegistry
from kronyx.viz.styling.default import DefaultSceneStyler

registry = IconRegistry()
registry.register("MyCustomLayer", Icon(id="my_custom", label="Custom", category="custom"))
styler = DefaultSceneStyler(icon_registry=registry)
```

## Extensibility

To add support for a new layer type:

1. Register an icon in `IconRegistry`
2. No renderer changes are required

The renderer consumes `StyledNode.icon` blindly and draws whatever icon id it receives.
