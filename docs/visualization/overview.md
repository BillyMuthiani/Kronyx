# Visualization Overview

Kronyx includes a native, renderer-independent visualization engine for generating model architecture diagrams.

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
SVG Output
```

## Components

| Component | Responsibility |
|:----------|:---------------|
| **GraphBuilder** | Inspects a Sequential model and produces an intermediate `Graph` |
| **Graph** | Immutable model structure (`Node`s and `Edge`s) |
| **LayoutEngine** | Computes node positions and canvas dimensions |
| **LayoutRegistry** | Plugin catalog for layout engines |
| **Scene** | Layout output with positioned nodes and edges |
| **SceneStyler** | Applies themes, fonts, colors, and icons to produce a `StyledScene` |
| **StyledScene** | Renderer-ready scene with all presentation properties applied |
| **RendererRegistry** | Plugin catalog for renderers |
| **ThemeRegistry** | Plugin catalog for visual themes |
| **IconRegistry** | Plugin catalog for layer icons |
| **SvgRenderer** | Paints a `StyledScene` into an SVG string |

## Usage

```python
from kronyx import Sequential, Dense, ReLU

model = Sequential()
model.add(Dense(2, 16))
model.add(ReLU())
model.add(Dense(16, 1))

model.visualize(output_format="svg")
```

This generates `model_architecture.svg` using the native visualization pipeline. No Graphviz is required.

### Graphviz Export

For non-SVG formats such as PNG, PDF, or DOT, Kronyx falls back to Graphviz-based export. Graphviz must be installed for these formats.

## Key Features

- Native SVG rendering without external dependencies
- Runtime theme switching (light, dark, blueprint, terminal, neon)
- Layer-specific semantic icons
- Registry-based plugin system
- Renderer-independent architecture
