# Visualization Engine Architecture

This document describes the internal architecture of the Kronyx visualization engine.

## Pipeline

```
Sequential Model
    ↓
GraphBuilder
    ↓
Graph
    ↓
Layout Engine
    ↓
Scene
    ↓
Scene Styler
    ↓
StyledScene
    ↓
Renderer
    ↓
Output (SVG, PNG, PDF, HTML, ASCII)
```

## Implemented

### Graph Layer

- **`Graph`**: Immutable container for model architecture. Contains `Node` and `Edge` objects.
- **`GraphBuilder`**: Inspects a `Sequential` model and produces a `Graph`. No coordinates, no rendering.
- **`Node`**: Represents a layer with `id`, `name`, `layer_type`, `params`, `output_shape`, `metadata`, and `tags`.
- **`Edge`**: Directed connection between two nodes.

### Layout Layer

- **`LayoutEngine`**: Abstract base class for layout algorithms.
- **`VerticalLayout`**: Default layout that arranges nodes in a single vertical column.
- **`LayoutRegistry`**: Central catalog for layout engines, allowing runtime lookup by name.
- **`Scene`**: Layout output containing positioned `PositionedNode`s and `PositionedEdge`s, plus canvas dimensions.

### Styling Layer

- **`SceneStyler`**: Abstract base class that transforms a `Scene` into a `StyledScene`.
- **`DefaultSceneStyler`**: Applies colors, fonts, borders, padding, and icons from `Theme` and `Style` dataclasses.
- **`StyledScene`**: Renderer-ready scene with all presentation properties applied.
- **`StyledNode`**: Node with presentation properties: `fill_color`, `stroke_color`, `font_family`, `icon`, etc.
- **`StyledEdge`**: Edge with presentation properties: `stroke_color`, `arrow_size`, `style`, etc.

### Theme Layer

- **`Theme`**: Immutable dataclass containing only appearance data (colors, fonts, sizes).
- **`ThemeRegistry`**: Central catalog for themes, allowing runtime lookup by name.
- Built-in themes: `LightTheme`, `DarkTheme`, `BlueprintTheme`, `TerminalTheme`, `NeonTheme`.

### Icon Layer

- **`Icon`**: Immutable semantic dataclass with `id`, `label`, and `category`.
- **`IconRegistry`**: Central catalog mapping layer types to icons.
- Built-in icons for Input, Output, Dense, Conv2D, Flatten, Dropout, BatchNormalization, ReLU, Sigmoid, Softmax, Tanh, LeakyReLU, MaxPooling2D, AveragePooling2D, and UnknownLayer.

### Renderer Layer

- **`Renderer`**: Abstract base class for all renderers.
- **`SvgRenderer`**: Native SVG renderer. Draws rectangles, text, edges, and icons from `StyledScene`.
- **`RendererRegistry`**: Central catalog for renderers.

### Engine

- **`VisualizationEngine`**: Coordinates the full pipeline.
  - Accepts optional `registry`, `layout_registry`, `theme_registry`, `icon_registry`, and `theme_name`.
  - Auto-registers built-in layouts, renderers, themes, and icons.
  - Default theme is `"light"`.

## Current Limitations

- Only one layout engine (`VerticalLayout`) is implemented.
- Only one renderer (`SvgRenderer`) is implemented.
- Themes control colors but do not support advanced effects (shadows, gradients, animations).
- Icons are represented as simple SVG primitives; no external icon fonts or images.
- No horizontal or grid layouts yet.

## Future Work

- Additional layout engines (horizontal, grid, nested)
- Additional renderers (PNG, PDF, HTML, ASCII)
- Advanced theme features (shadows, gradients, glow effects)
- Icon customization per theme
- Animation and interactive rendering
- Educational overlays (parameter counts, shape inference)
