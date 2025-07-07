import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bio_snn.predictive_coding import PredictiveCodingNetwork

"""
Animated visualization of the spiking neural network in real time.

This script runs a small PredictiveCodingNetwork and displays membrane potentials, thresholds, and spike activity for each neuron.
"""


def run_simulation(steps=200):
    """Run a live simulation of a small predictive coding network."""
    net = PredictiveCodingNetwork([2, 3, 1])

    # storage for neuron states
    voltages = [[[] for _ in layer.neurons] for layer in net.layers]
    thresholds = [[[] for _ in layer.neurons] for layer in net.layers]
    spikes = [[np.zeros(steps) for _ in layer.neurons] for layer in net.layers]

    fig, axes = plt.subplots(len(net.layers), 2, figsize=(10, 5 * len(net.layers)))
    if len(net.layers) == 1:
        axes = np.array([axes])  # ensure 2D indexing

    lines = []
    spike_imgs = []
    for i, layer in enumerate(net.layers):
        ax_v = axes[i, 0]
        ax_s = axes[i, 1]

        layer_lines = []
        for j, _ in enumerate(layer.neurons):
            (line_v,) = ax_v.plot([], [], label=f"neuron {j} V")
            (line_t,) = ax_v.plot([], [], '--', label=f"neuron {j} Th")
            layer_lines.append((line_v, line_t))
        ax_v.set_xlim(0, steps)
        ax_v.set_ylim(-0.5, 1.5)
        ax_v.set_xlabel('step')
        ax_v.set_ylabel('voltage')
        ax_v.set_title(f'Layer {i} membrane potential')
        ax_v.legend(loc='upper right')

        img = ax_s.imshow(
            np.zeros((len(layer.neurons), steps)),
            aspect='auto',
            cmap='Greys',
            origin='lower',
            interpolation='nearest',
            extent=[0, steps, 0, len(layer.neurons)],
        )
        ax_s.set_xlabel('step')
        ax_s.set_ylabel('neuron')
        ax_s.set_title(f'Layer {i} spikes')
        layer_lines.append(img)

        lines.append(layer_lines)
        spike_imgs.append(img)

    def update(frame):
        x = np.array([np.sin(frame / 10.0), np.cos(frame / 15.0)])
        activations = [x]
        for i, layer in enumerate(net.layers):
            prev = activations[-1]
            out = []
            for j, neuron in enumerate(layer.neurons):
                inp = layer.weights[j] * prev
                spike = neuron.forward(inp)
                out.append(spike)
                voltages[i][j].append(neuron.v)
                thresholds[i][j].append(neuron.threshold)
                spikes[i][j][frame] = spike

                for k, pre in enumerate(prev):
                    layer.weights[j, k], layer.pre_traces[j, k], layer.post_traces[j] = layer.stdp.update(
                        layer.weights[j, k],
                        pre,
                        spike,
                        layer.pre_traces[j, k],
                        layer.post_traces[j],
                        modulation=1.0,
                    )
                layer.avg_rates[j] += (spike - layer.avg_rates[j]) / layer.tau_rate
                layer.weights[j] = layer.scaling.update(layer.weights[j], layer.avg_rates[j])
            out = np.array(out)
            pred = net.pred_weights[i].T @ prev
            err = out - pred
            net.pred_weights[i] += net.lr * np.outer(prev, err)
            activations.append(err)

        artist_list = []
        for li, layer_lines in enumerate(lines):
            for ni, pair in enumerate(layer_lines[:-1]):
                pair[0].set_data(range(len(voltages[li][ni])), voltages[li][ni])
                pair[1].set_data(range(len(thresholds[li][ni])), thresholds[li][ni])
                artist_list.extend(pair)
            spike_imgs[li].set_data(spikes[li])
            artist_list.append(spike_imgs[li])
        return artist_list

    ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
