import itertools
import matplotlib.pyplot as plt
import torch

# This is just a helper file for the main code.
# No need to make changes in this file. 

# Visualization function for losses and parameters
def visualize_training(epochs_range, losses, parameter_evolution, save_filename=None, save_format='pdf'):
    """
    Visualize the training process by plotting the loss curve and parameter evolution over epochs.

    Parameters:
    epochs_range (iterable): A range or list of epoch indices.
    losses (list or iterable): A list of loss values recorded over epochs.
    parameter_evolution (dict): A dictionary containing the evolution of parameters over epochs.
                                Keys should be parameter names and values should be lists of torch tensors.
    save_filename (str, optional): Base filename for saving plots. If provided, the plots will be saved with
                                   '_loss' and '_params' suffixes in the specified format.
    save_format (str, optional): Format in which to save the plots (e.g., 'pdf', 'png'). Default is 'pdf'.

    Returns:
    None
    """
    plt.style.use('fivethirtyeight')

    # Define line styles for better distinction
    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]  # 5 unique styles
    line_styles = itertools.cycle(line_styles)
    linewidth = 4

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, losses, label="Loss", linewidth=2)  # Increased line thickness
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)  # Added mild grid lines
    plt.legend(handlelength=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if save_filename:
        plt.savefig(f"{save_filename}_loss.{save_format}", format=save_format)
    plt.show()

    # Plot parameter evolution
    plt.figure(figsize=(12, 8))
    for param_name, param_values in parameter_evolution.items():
        param_values = torch.stack(param_values).numpy()
        if param_values.ndim == 2:  # For matrices like W0, W1
            for i in range(param_values.shape[1]):
                plt.plot(
                    epochs_range, 
                    param_values[:, i], 
                    linestyle=next(line_styles), 
                    linewidth=linewidth,
                    label=f"{param_name}[{i}]"
                )
        elif param_values.ndim == 3:  # For tensors like W1 with 2D arrays
            for i in range(param_values.shape[1]):
                for j in range(param_values.shape[2]):
                    plt.plot(
                        epochs_range, 
                        param_values[:, i, j], 
                        linestyle=next(line_styles), 
                        linewidth=linewidth,
                        label=f"{param_name}[{i},{j}]"
                    )
        else:  # For 1D arrays like W2, b0, b1, b2
            plt.plot(
                epochs_range, 
                param_values.flatten(), 
                linestyle=next(line_styles), 
                linewidth=linewidth,  # Increased line thickness
                label=f"{param_name}"
            )

    # Adjust plot aesthetics
    plt.title("Parameter Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Parameter Values")
    plt.grid(alpha=0.3)  # Added mild grid lines
    plt.legend(handlelength=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if save_filename:
        plt.savefig(f"{save_filename}_params.{save_format}", format=save_format)
    plt.show()


# Function to generate a complete LaTeX code for including the saved plots in a standard document
def generate_tex_code(base_filename, save_format='pdf'):
    """
    Generate complete LaTeX code to include two figures (loss and parameter evolution) side by side in a LaTeX document.

    Parameters:
    base_filename (str): Base filename used for the saved plots.
    save_format (str, optional): Format of the saved plots (e.g., 'pdf', 'png'). Default is 'pdf'.

    Returns:
    str: A string containing the complete LaTeX code to include both plots.
    """
    tex_code = (
rf"""\documentclass{{article}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage[letterpaper, margin=1in]{{geometry}}  % Optional, for better page layout
\usepackage{{subcaption}}
\begin{{document}}

\title{{Training Plots Report}}
\author{{}}
\date{{}}
\maketitle

\begin{{figure}}[ht]
    \centering
    \begin{{subfigure}}[b]{{0.45\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{base_filename}_loss.{save_format}}}
        \caption{{Loss Curve}}
        \label{{fig:{base_filename}_loss}}
    \end{{subfigure}}
    \hfill
    \begin{{subfigure}}[b]{{0.45\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{base_filename}_params.{save_format}}}
        \caption{{Parameter Evolution}}
        \label{{fig:{base_filename}_params}}
    \end{{subfigure}}
    \caption{{Training Plots Report}}
    \label{{fig:training_plots}}
\end{{figure}}

\end{{document}}
""")
    return tex_code
