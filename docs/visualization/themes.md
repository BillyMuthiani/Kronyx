# Themes

Themes are immutable dataclasses that control the visual appearance of every element in a diagram. They are renderer-independent and shared across all output formats.

## Built-in Themes

| Theme | Description |
|:------|:------------|
| **light** | Default white background with black text |
| **dark** | Dark gray background with light text |
| **blueprint** | Engineering blueprint style with blue tones |
| **terminal** | Monochrome green on black, Courier New font |
| **neon** | High-contrast neon colors on black |

## Usage

```python
from kronyx import Sequential, Dense
from kronyx.viz.themes import DarkTheme, ThemeRegistry
from kronyx.viz.styling.default import DefaultSceneStyler
from kronyx.viz.engine import VisualizationEngine

model = Sequential()
model.add(Dense(2, 4))

# Via engine theme_name parameter
engine = VisualizationEngine(theme_name="dark")
engine.visualize(model, renderer_name="svg")

# Or via direct styler construction
theme_registry = ThemeRegistry()
theme_registry.register("my-theme", DarkTheme)
styler = DefaultSceneStyler(theme=DarkTheme)
engine = VisualizationEngine(styler=styler)
engine.visualize(model, renderer_name="svg")
```

## Theme Fields

Themes contain only appearance data. No renderer logic, no XML, no SVG.

| Category | Fields |
|:---------|:-------|
| General | `background_color`, `text_color`, `secondary_text_color` |
| Node | `node_fill`, `node_border`, `node_border_width`, `node_radius` |
| Edges | `edge_color`, `edge_width`, `arrow_color` |
| Typography | `title_font_family`, `body_font_family`, `title_font_size`, `body_font_size` |
| Metadata | `metadata_color` |
| Canvas | `canvas_background` |
| Optional | `shadow_color`, `grid_color`, `selection_color` |

## Custom Themes

Create a custom theme by instantiating `Theme` with your desired values:

```python
from kronyx.viz.themes import Theme

custom_theme = Theme(
    background_color="#1a1a1a",
    text_color="#e0e0e0",
    node_fill="#2d2d2d",
    node_border="#555555",
    edge_color="#777777",
)
```

Themes are frozen dataclasses and cannot be modified after creation.
