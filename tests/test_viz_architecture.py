"""Architecture tests for the Kronyx visualization engine Phase 1."""

from kronyx import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    ReLU,
    Sequential,
)
from kronyx.viz.builder import GraphBuilder
from kronyx.viz.graph import Edge, Graph, Node


class TestGraphRepr:
    """Verify concise Graph representation."""

    def test_repr_shows_counts(self):
        graph = Graph(
            nodes=[Node(id="n1", name="Dense", layer_type="Dense", params=10)],
            edges=[Edge(source="input", target="n1")],
        )
        r = repr(graph)
        assert "nodes=1" in r
        assert "edges=1" in r

    def test_repr_excludes_full_node_list(self):
        graph = Graph(
            nodes=[Node(id=f"n{i}", name="Dense", layer_type="Dense", params=i) for i in range(10)],
            edges=[],
        )
        r = repr(graph)
        assert "Node(" not in r

    def test_repr_includes_metadata_when_present(self):
        graph = Graph(
            nodes=[Node(id="n1", name="Dense", layer_type="Dense", params=10)],
            edges=[],
            metadata={"total_params": 10},
        )
        r = repr(graph)
        assert "metadata" in r


class TestGraphProperties:
    """Verify read-only Graph statistics properties."""

    def test_num_nodes(self):
        graph = Graph(
            nodes=[Node(id="n1", name="A", layer_type="Dense", params=1),
                   Node(id="n2", name="B", layer_type="ReLU", params=0)],
            edges=[Edge(source="n1", target="n2")],
        )
        assert graph.num_nodes == 2

    def test_num_edges(self):
        graph = Graph(
            nodes=[Node(id="n1", name="A", layer_type="Dense", params=1)],
            edges=[Edge(source="n1", target="n2") for n2 in ("n2", "n3", "n4")],
        )
        assert graph.num_edges == 3

    def test_num_trainable_layers(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=10, tags={"trainable"}),
                Node(id="n2", name="B", layer_type="ReLU", params=0, tags={"activation"}),
                Node(id="n3", name="C", layer_type="Dense", params=5, tags={"trainable"}),
            ],
            edges=[],
        )
        assert graph.num_trainable_layers == 2

    def test_num_activation_layers(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=10, tags={"trainable"}),
                Node(id="n2", name="B", layer_type="ReLU", params=0, tags={"activation"}),
                Node(id="n3", name="C", layer_type="Sigmoid", params=0, tags={"activation"}),
            ],
            edges=[],
        )
        assert graph.num_activation_layers == 2

    def test_total_params(self):
        graph = Graph(
            nodes=[
                Node(id="n1", name="A", layer_type="Dense", params=10, tags={"trainable"}),
                Node(id="n2", name="B", layer_type="ReLU", params=0, tags={"activation"}),
            ],
            edges=[],
        )
        assert graph.total_params == 10


class TestGraphBuilderMetadata:
    """Verify Node metadata is populated correctly."""

    def test_dense_metadata(self):
        model = Sequential()
        model.add(Dense(2, 8))
        graph = GraphBuilder().build(model)
        dense_nodes = [n for n in graph.nodes if n.layer_type == "Dense"]
        assert len(dense_nodes) == 1
        assert dense_nodes[0].metadata.get("units") == 8

    def test_conv2d_metadata(self):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=3, padding="valid"))
        graph = GraphBuilder().build(model)
        conv_nodes = [n for n in graph.nodes if n.layer_type == "Conv2D"]
        assert len(conv_nodes) == 1
        meta = conv_nodes[0].metadata
        assert meta.get("filters") == 16
        assert meta.get("kernel_size") == 3
        assert meta.get("padding") == "valid"

    def test_dropout_metadata(self):
        model = Sequential()
        model.add(Dropout(0.25))
        graph = GraphBuilder().build(model)
        dropout_nodes = [n for n in graph.nodes if n.layer_type == "Dropout"]
        assert len(dropout_nodes) == 1
        assert dropout_nodes[0].metadata.get("rate") == 0.25

    def test_batchnorm_metadata(self):
        model = Sequential()
        model.add(BatchNormalization())
        graph = GraphBuilder().build(model)
        bn_nodes = [n for n in graph.nodes if n.layer_type == "BatchNormalization"]
        assert len(bn_nodes) == 1
        assert "momentum" in bn_nodes[0].metadata
        assert "epsilon" in bn_nodes[0].metadata

    def test_activation_metadata(self):
        model = Sequential()
        model.add(ReLU())
        graph = GraphBuilder().build(model)
        act_nodes = [n for n in graph.nodes if n.layer_type == "ReLU"]
        assert len(act_nodes) == 1
        assert act_nodes[0].metadata.get("activation") == "ReLU"

    def test_input_output_metadata_empty(self):
        model = Sequential()
        model.add(Dense(2, 4))
        graph = GraphBuilder().build(model)
        input_nodes = [n for n in graph.nodes if n.layer_type == "Input"]
        output_nodes = [n for n in graph.nodes if n.layer_type == "Output"]
        assert len(input_nodes) == 1
        assert len(output_nodes) == 1
        assert input_nodes[0].metadata == {}
        assert output_nodes[0].metadata == {}


class TestGraphBuilderTags:
    """Verify Node tags are assigned correctly."""

    def test_dense_tag(self):
        model = Sequential()
        model.add(Dense(2, 4))
        graph = GraphBuilder().build(model)
        dense_nodes = [n for n in graph.nodes if n.layer_type == "Dense"]
        assert "dense" in dense_nodes[0].tags

    def test_conv2d_tag(self):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=3))
        graph = GraphBuilder().build(model)
        conv_nodes = [n for n in graph.nodes if n.layer_type == "Conv2D"]
        assert "convolution" in conv_nodes[0].tags

    def test_activation_tag(self):
        model = Sequential()
        model.add(ReLU())
        graph = GraphBuilder().build(model)
        act_nodes = [n for n in graph.nodes if n.layer_type == "ReLU"]
        assert "activation" in act_nodes[0].tags

    def test_dropout_tag(self):
        model = Sequential()
        model.add(Dropout(0.1))
        graph = GraphBuilder().build(model)
        assert "regularization" in graph.nodes[1].tags

    def test_batchnorm_tag(self):
        model = Sequential()
        model.add(BatchNormalization())
        graph = GraphBuilder().build(model)
        assert "normalization" in graph.nodes[1].tags

    def test_trainable_tag(self):
        model = Sequential()
        model.add(Dense(2, 4))
        graph = GraphBuilder().build(model)
        dense_nodes = [n for n in graph.nodes if n.layer_type == "Dense"]
        assert "trainable" in dense_nodes[0].tags

    def test_input_output_tags(self):
        model = Sequential()
        model.add(Dense(2, 4))
        graph = GraphBuilder().build(model)
        assert "input" in graph.nodes[0].tags
        assert "output" in graph.nodes[-1].tags


class TestGraphBuilderStatistics:
    """Verify Graph statistics match built output."""

    def test_statistics_counts(self):
        model = Sequential()
        model.add(Dense(2, 8))
        model.add(ReLU())
        model.add(Dense(8, 2))
        graph = GraphBuilder().build(model)
        assert graph.num_nodes == 5
        assert graph.num_edges == 4
        assert graph.num_trainable_layers == 2
        assert graph.num_activation_layers == 1
        assert graph.total_params == (2 * 8 + 8) + (8 * 2 + 2)
