import click
from tftrt.examples.object_detection import optimize_model

@click.command()
@click.argument("config_path")
@click.argument("checkpoint")
@click.argument("output_graph")
def main(config_path, checkpoint, output_graph):
    frozen_graph = optimize_model(
        config_path=config_path,
        checkpoint_path=checkpoint,
        use_trt=True,
        precision_mode='FP16'
    )
    with open(output_graph, 'wb') as f:
        f.write(frozen_graph.SerializeToString())

if __name__ == "__main__":
    main()
