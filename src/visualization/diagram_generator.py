"""
Conceptual diagrams and flowcharts generator for PINN tutorial system.

Provides tools for creating neural network architecture diagrams,
physics equation flowcharts, and data flow visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

from ..core.data_models import ModelConfig, TrainingConfig


class DiagramGenerator:
    """
    Generator for conceptual diagrams and flowcharts.
    
    Provides methods for creating neural network architecture diagrams,
    physics equation flowcharts, loss component visualizations, and
    data flow diagrams for the PINN training process.
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the diagram generator.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'input': '#E8F4FD',
            'hidden': '#B3D9FF',
            'output': '#4A90E2',
            'physics': '#FF6B6B',
            'data': '#4ECDC4',
            'loss': '#FFE66D',
            'arrow': '#2C3E50',
            'text': '#2C3E50',
            'border': '#34495E'
        }
        self.setup_style()
    
    def setup_style(self):
        """Set up consistent plotting style for diagrams."""
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'font.size': 10,
            'axes.titlesize': 14,
            'figure.titlesize': 16,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.bottom': False,
            'axes.spines.left': False,
            'xtick.bottom': False,
            'ytick.left': False,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def create_pinn_architecture_diagram(self, 
                                       model_config: ModelConfig,
                                       title: str = "PINN Architecture",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a neural network architecture diagram for PINN.
        
        Args:
            model_config: ModelConfig object with architecture details
            title: Diagram title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Define layer positions and sizes
        input_features = model_config.input_features
        hidden_layers = model_config.hidden_layers
        output_features = model_config.output_features
        
        n_input = len(input_features)
        n_output = len(output_features)
        n_hidden_layers = len(hidden_layers)
        
        # Calculate positions
        layer_spacing = 2.5
        total_width = (n_hidden_layers + 2) * layer_spacing
        
        # Input layer
        input_x = 0.5
        input_y_positions = np.linspace(0.2, 0.8, n_input)
        
        # Hidden layers
        hidden_x_positions = [input_x + (i + 1) * layer_spacing for i in range(n_hidden_layers)]
        
        # Output layer
        output_x = input_x + (n_hidden_layers + 1) * layer_spacing
        output_y_positions = np.linspace(0.3, 0.7, n_output)
        
        # Draw input layer
        input_nodes = []
        for i, (feature, y_pos) in enumerate(zip(input_features, input_y_positions)):
            circle = Circle((input_x, y_pos), 0.08, color=self.colors['input'], 
                          ec=self.colors['border'], linewidth=2)
            ax.add_patch(circle)
            input_nodes.append((input_x, y_pos))
            
            # Add feature labels
            ax.text(input_x - 0.3, y_pos, feature, ha='right', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Add input layer label
        ax.text(input_x, 0.05, 'Input Layer', ha='center', va='center', 
               fontsize=12, fontweight='bold', color=self.colors['text'])
        
        # Draw hidden layers
        hidden_nodes = []
        for layer_idx, (x_pos, n_neurons) in enumerate(zip(hidden_x_positions, hidden_layers)):
            layer_nodes = []
            y_positions = np.linspace(0.15, 0.85, n_neurons)
            
            for y_pos in y_positions:
                circle = Circle((x_pos, y_pos), 0.06, color=self.colors['hidden'], 
                              ec=self.colors['border'], linewidth=1.5)
                ax.add_patch(circle)
                layer_nodes.append((x_pos, y_pos))
            
            hidden_nodes.append(layer_nodes)
            
            # Add layer label
            ax.text(x_pos, 0.05, f'Hidden {layer_idx + 1}\n({n_neurons} neurons)', 
                   ha='center', va='center', fontsize=10, fontweight='bold', 
                   color=self.colors['text'])
        
        # Draw output layer
        output_nodes = []
        for i, (feature, y_pos) in enumerate(zip(output_features, output_y_positions)):
            circle = Circle((output_x, y_pos), 0.08, color=self.colors['output'], 
                          ec=self.colors['border'], linewidth=2)
            ax.add_patch(circle)
            output_nodes.append((output_x, y_pos))
            
            # Add feature labels
            ax.text(output_x + 0.3, y_pos, feature, ha='left', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Add output layer label
        ax.text(output_x, 0.05, 'Output Layer', ha='center', va='center', 
               fontsize=12, fontweight='bold', color=self.colors['text'])
        
        # Draw connections (sample connections to avoid clutter)
        def draw_sample_connections(from_nodes, to_nodes, alpha=0.3, color='gray'):
            """Draw sample connections between layers."""
            n_connections = min(len(from_nodes) * len(to_nodes), 20)  # Limit connections
            
            for i in range(0, len(from_nodes), max(1, len(from_nodes) // 4)):
                for j in range(0, len(to_nodes), max(1, len(to_nodes) // 4)):
                    x1, y1 = from_nodes[i]
                    x2, y2 = to_nodes[j]
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)
        
        # Connect input to first hidden layer
        if hidden_nodes:
            draw_sample_connections(input_nodes, hidden_nodes[0])
        
        # Connect hidden layers
        for i in range(len(hidden_nodes) - 1):
            draw_sample_connections(hidden_nodes[i], hidden_nodes[i + 1])
        
        # Connect last hidden layer to output
        if hidden_nodes:
            draw_sample_connections(hidden_nodes[-1], output_nodes)
        else:
            # Direct connection if no hidden layers
            draw_sample_connections(input_nodes, output_nodes)
        
        # Add activation function annotation
        if hasattr(model_config, 'activation_function'):
            activation = model_config.activation_function
            ax.text(0.5, 0.95, f'Activation Function: {activation.upper()}', 
                   transform=ax.transAxes, ha='center', va='top', 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        # Add physics-informed components
        physics_box = FancyBboxPatch((output_x + 0.8, 0.4), 1.5, 0.3, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=self.colors['physics'], 
                                   edgecolor=self.colors['border'], 
                                   linewidth=2, alpha=0.8)
        ax.add_patch(physics_box)
        
        ax.text(output_x + 1.55, 0.55, 'Physics-Informed\nLoss Components', 
               ha='center', va='center', fontsize=11, fontweight='bold', 
               color='white')
        
        # Add arrows from output to physics components
        for x_out, y_out in output_nodes:
            arrow = patches.FancyArrowPatch((x_out + 0.1, y_out), 
                                          (output_x + 0.8, 0.55),
                                          arrowstyle='->', mutation_scale=20, 
                                          color=self.colors['arrow'], linewidth=2)
            ax.add_patch(arrow)
        
        # Set axis properties
        ax.set_xlim(-0.5, output_x + 2.8)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def create_physics_equations_flowchart(self, 
                                         equations: List[str] = None,
                                         title: str = "Physics Equations in PINN",
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create flowchart showing physics equations and their relationships.
        
        Args:
            equations: List of equation names (default: standard reservoir equations)
            title: Flowchart title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if equations is None:
            equations = ["Darcy's Law", "Buckley-Leverett", "Continuity Equation"]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define equation details
        equation_details = {
            "Darcy's Law": {
                'formula': r"$\nabla \cdot \left(\frac{k}{\mu} \nabla p\right) = 0$",
                'description': 'Fluid flow in porous media',
                'color': '#FF6B6B'
            },
            "Buckley-Leverett": {
                'formula': r"$\frac{\partial S}{\partial t} + \frac{\partial f(S)}{\partial x} = 0$",
                'description': 'Two-phase flow saturation',
                'color': '#4ECDC4'
            },
            "Continuity Equation": {
                'formula': r"$\nabla \cdot \vec{v} = 0$",
                'description': 'Mass conservation',
                'color': '#45B7D1'
            }
        }
        
        # Position equations
        n_equations = len(equations)
        positions = {
            "Darcy's Law": (0.2, 0.7),
            "Buckley-Leverett": (0.8, 0.7),
            "Continuity Equation": (0.5, 0.3)
        }
        
        # Draw equation boxes
        equation_boxes = {}
        for eq_name in equations:
            if eq_name in equation_details and eq_name in positions:
                x, y = positions[eq_name]
                details = equation_details[eq_name]
                
                # Create equation box
                box = FancyBboxPatch((x - 0.15, y - 0.1), 0.3, 0.2, 
                                   boxstyle="round,pad=0.02", 
                                   facecolor=details['color'], 
                                   edgecolor=self.colors['border'], 
                                   linewidth=2, alpha=0.8)
                ax.add_patch(box)
                equation_boxes[eq_name] = (x, y)
                
                # Add equation formula
                ax.text(x, y + 0.05, details['formula'], ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
                
                # Add description
                ax.text(x, y - 0.05, details['description'], ha='center', va='center', 
                       fontsize=9, color='white', style='italic')
                
                # Add equation name
                ax.text(x, y + 0.15, eq_name, ha='center', va='center', 
                       fontsize=11, fontweight='bold', color=self.colors['text'])
        
        # Draw relationships between equations
        relationships = [
            ("Darcy's Law", "Continuity Equation", "Velocity field"),
            ("Buckley-Leverett", "Continuity Equation", "Saturation transport"),
            ("Darcy's Law", "Buckley-Leverett", "Pressure-saturation coupling")
        ]
        
        for eq1, eq2, relationship in relationships:
            if eq1 in equation_boxes and eq2 in equation_boxes:
                x1, y1 = equation_boxes[eq1]
                x2, y2 = equation_boxes[eq2]
                
                # Draw arrow
                arrow = patches.FancyArrowPatch((x1, y1 - 0.1), (x2, y2 + 0.1),
                                              arrowstyle='<->', mutation_scale=20, 
                                              color=self.colors['arrow'], linewidth=2)
                ax.add_patch(arrow)
                
                # Add relationship label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, relationship, ha='center', va='center', 
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.8))
        
        # Add PINN integration box
        pinn_box = FancyBboxPatch((0.35, 0.05), 0.3, 0.1, 
                                boxstyle="round,pad=0.02", 
                                facecolor=self.colors['loss'], 
                                edgecolor=self.colors['border'], 
                                linewidth=2, alpha=0.9)
        ax.add_patch(pinn_box)
        
        ax.text(0.5, 0.1, 'PINN Loss Function\nIntegration', ha='center', va='center', 
               fontsize=11, fontweight='bold', color=self.colors['text'])
        
        # Draw arrows from equations to PINN
        for eq_name, (x, y) in equation_boxes.items():
            arrow = patches.FancyArrowPatch((x, y - 0.1), (0.5, 0.15),
                                          arrowstyle='->', mutation_scale=15, 
                                          color=self.colors['arrow'], linewidth=1.5,
                                          alpha=0.7)
            ax.add_patch(arrow)
        
        # Add legend for equation types
        legend_elements = []
        for eq_name, details in equation_details.items():
            if eq_name in equations:
                legend_elements.append(patches.Patch(color=details['color'], 
                                                   label=eq_name))
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98))
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def create_loss_components_diagram(self, 
                                     loss_weights: Dict[str, float] = None,
                                     title: str = "PINN Loss Components",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create diagram showing loss function components and their weights.
        
        Args:
            loss_weights: Dictionary of loss component weights
            title: Diagram title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if loss_weights is None:
            loss_weights = {'data': 1.0, 'pde': 1.0, 'boundary': 1.0}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Loss component structure
        ax = ax1
        
        # Define loss components
        components = {
            'Data Loss': {
                'formula': r'$\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^{N} ||u_{\theta}(x_i) - u_i||^2$',
                'description': 'Supervised learning\nfrom measurements',
                'color': self.colors['data'],
                'position': (0.2, 0.8)
            },
            'PDE Loss': {
                'formula': r'$\mathcal{L}_{PDE} = \frac{1}{M} \sum_{j=1}^{M} ||f(x_j, u_{\theta}, \nabla u_{\theta})||^2$',
                'description': 'Physics equation\nresiduals',
                'color': self.colors['physics'],
                'position': (0.2, 0.5)
            },
            'Boundary Loss': {
                'formula': r'$\mathcal{L}_{BC} = \frac{1}{K} \sum_{k=1}^{K} ||B(x_k, u_{\theta})||^2$',
                'description': 'Boundary condition\nenforcement',
                'color': '#9B59B6',
                'position': (0.2, 0.2)
            }
        }
        
        # Draw component boxes
        component_positions = {}
        for comp_name, details in components.items():
            x, y = details['position']
            
            # Create component box
            box = FancyBboxPatch((x - 0.15, y - 0.08), 0.3, 0.16, 
                               boxstyle="round,pad=0.02", 
                               facecolor=details['color'], 
                               edgecolor=self.colors['border'], 
                               linewidth=2, alpha=0.8)
            ax.add_patch(box)
            component_positions[comp_name] = (x, y)
            
            # Add component name
            ax.text(x, y + 0.05, comp_name, ha='center', va='center', 
                   fontsize=11, fontweight='bold', color='white')
            
            # Add description
            ax.text(x, y - 0.03, details['description'], ha='center', va='center', 
                   fontsize=9, color='white', style='italic')
            
            # Add formula (to the right)
            ax.text(x + 0.25, y, details['formula'], ha='left', va='center', 
                   fontsize=10, color=self.colors['text'])
            
            # Add weight
            weight = loss_weights.get(comp_name.lower().split()[0], 1.0)
            ax.text(x - 0.2, y, f'λ = {weight}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=self.colors['text'],
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Draw total loss combination
        total_box = FancyBboxPatch((0.7, 0.45), 0.25, 0.1, 
                                 boxstyle="round,pad=0.02", 
                                 facecolor=self.colors['loss'], 
                                 edgecolor=self.colors['border'], 
                                 linewidth=3, alpha=0.9)
        ax.add_patch(total_box)
        
        ax.text(0.825, 0.5, 'Total Loss', ha='center', va='center', 
               fontsize=12, fontweight='bold', color=self.colors['text'])
        
        # Add total loss formula
        total_formula = r'$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{PDE} + \lambda_3 \mathcal{L}_{BC}$'
        ax.text(0.5, 0.05, total_formula, ha='center', va='center', 
               fontsize=12, fontweight='bold', color=self.colors['text'],
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Draw arrows to total loss
        for comp_name, (x, y) in component_positions.items():
            arrow = patches.FancyArrowPatch((x + 0.15, y), (0.7, 0.5),
                                          arrowstyle='->', mutation_scale=15, 
                                          color=self.colors['arrow'], linewidth=2)
            ax.add_patch(arrow)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Loss Function Components', fontsize=14, fontweight='bold')
        
        # Right plot: Loss weight visualization
        ax = ax2
        
        # Pie chart of loss weights
        labels = list(loss_weights.keys())
        weights = list(loss_weights.values())
        colors = [self.colors['data'], self.colors['physics'], '#9B59B6'][:len(weights)]
        
        wedges, texts, autotexts = ax.pie(weights, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Enhance pie chart appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Loss Weight Distribution', fontsize=14, fontweight='bold')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def create_training_process_flowchart(self, 
                                        config: TrainingConfig = None,
                                        title: str = "PINN Training Process",
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create flowchart showing the complete PINN training process.
        
        Args:
            config: TrainingConfig object with training parameters
            title: Flowchart title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 16))
        
        # Define process steps
        steps = [
            {
                'name': 'Data Preparation',
                'description': 'Load and preprocess\nLAS well log data',
                'position': (0.5, 0.95),
                'color': self.colors['data'],
                'size': (0.25, 0.06)
            },
            {
                'name': 'Model Initialization',
                'description': 'Initialize PINN\narchitecture',
                'position': (0.5, 0.85),
                'color': self.colors['input'],
                'size': (0.25, 0.06)
            },
            {
                'name': 'Forward Pass',
                'description': 'Compute predictions\nu_θ(x)',
                'position': (0.2, 0.75),
                'color': self.colors['hidden'],
                'size': (0.2, 0.05)
            },
            {
                'name': 'Physics Evaluation',
                'description': 'Compute PDE residuals\nf(x, u_θ, ∇u_θ)',
                'position': (0.8, 0.75),
                'color': self.colors['physics'],
                'size': (0.2, 0.05)
            },
            {
                'name': 'Loss Computation',
                'description': 'Combine data, PDE,\nand boundary losses',
                'position': (0.5, 0.65),
                'color': self.colors['loss'],
                'size': (0.25, 0.06)
            },
            {
                'name': 'Backpropagation',
                'description': 'Compute gradients\n∇_θ L',
                'position': (0.5, 0.55),
                'color': '#E74C3C',
                'size': (0.2, 0.05)
            },
            {
                'name': 'Adam Optimizer',
                'description': f'Update parameters\n(Epochs 1-{config.optimizer_switch_epoch if config else 3000})',
                'position': (0.25, 0.45),
                'color': '#3498DB',
                'size': (0.2, 0.05)
            },
            {
                'name': 'L-BFGS Optimizer',
                'description': f'Refine parameters\n(Epochs {config.optimizer_switch_epoch + 1 if config else 3001}-{config.num_epochs if config else 5000})',
                'position': (0.75, 0.45),
                'color': '#2ECC71',
                'size': (0.2, 0.05)
            },
            {
                'name': 'Convergence Check',
                'description': 'Check stopping\ncriteria',
                'position': (0.5, 0.35),
                'color': '#F39C12',
                'size': (0.2, 0.05)
            },
            {
                'name': 'Validation',
                'description': 'Evaluate on\nheld-out data',
                'position': (0.5, 0.25),
                'color': '#9B59B6',
                'size': (0.2, 0.05)
            },
            {
                'name': 'Results Analysis',
                'description': 'Generate predictions\nand error analysis',
                'position': (0.5, 0.15),
                'color': '#1ABC9C',
                'size': (0.25, 0.06)
            }
        ]
        
        # Draw process boxes
        step_positions = {}
        for step in steps:
            x, y = step['position']
            w, h = step['size']
            
            # Create step box
            box = FancyBboxPatch((x - w/2, y - h/2), w, h, 
                               boxstyle="round,pad=0.01", 
                               facecolor=step['color'], 
                               edgecolor=self.colors['border'], 
                               linewidth=2, alpha=0.8)
            ax.add_patch(box)
            step_positions[step['name']] = (x, y)
            
            # Add step name
            ax.text(x, y + 0.015, step['name'], ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
            
            # Add description
            ax.text(x, y - 0.015, step['description'], ha='center', va='center', 
                   fontsize=8, color='white', style='italic')
        
        # Define connections between steps
        connections = [
            ('Data Preparation', 'Model Initialization'),
            ('Model Initialization', 'Forward Pass'),
            ('Model Initialization', 'Physics Evaluation'),
            ('Forward Pass', 'Loss Computation'),
            ('Physics Evaluation', 'Loss Computation'),
            ('Loss Computation', 'Backpropagation'),
            ('Backpropagation', 'Adam Optimizer'),
            ('Backpropagation', 'L-BFGS Optimizer'),
            ('Adam Optimizer', 'Convergence Check'),
            ('L-BFGS Optimizer', 'Convergence Check'),
            ('Convergence Check', 'Validation'),
            ('Validation', 'Results Analysis')
        ]
        
        # Draw connections
        for start_step, end_step in connections:
            if start_step in step_positions and end_step in step_positions:
                x1, y1 = step_positions[start_step]
                x2, y2 = step_positions[end_step]
                
                # Calculate connection points
                if abs(x1 - x2) > abs(y1 - y2):  # Horizontal connection
                    if x1 < x2:
                        start_point = (x1 + 0.125, y1)
                        end_point = (x2 - 0.125, y2)
                    else:
                        start_point = (x1 - 0.125, y1)
                        end_point = (x2 + 0.125, y2)
                else:  # Vertical connection
                    if y1 > y2:
                        start_point = (x1, y1 - 0.03)
                        end_point = (x2, y2 + 0.03)
                    else:
                        start_point = (x1, y1 + 0.03)
                        end_point = (x2, y2 - 0.03)
                
                arrow = patches.FancyArrowPatch(start_point, end_point,
                                              arrowstyle='->', mutation_scale=15, 
                                              color=self.colors['arrow'], linewidth=2)
                ax.add_patch(arrow)
        
        # Add feedback loop for convergence check
        feedback_arrow = patches.FancyArrowPatch((0.4, 0.35), (0.1, 0.75),
                                               arrowstyle='->', mutation_scale=15, 
                                               color='red', linewidth=2, linestyle='--',
                                               connectionstyle="arc3,rad=0.3")
        ax.add_patch(feedback_arrow)
        
        ax.text(0.05, 0.55, 'Continue\nTraining', ha='center', va='center', 
               fontsize=9, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add decision diamond for convergence
        diamond = patches.RegularPolygon((0.5, 0.35), 4, radius=0.04, 
                                       orientation=np.pi/4, 
                                       facecolor='yellow', 
                                       edgecolor=self.colors['border'], 
                                       linewidth=2)
        ax.add_patch(diamond)
        
        ax.text(0.55, 0.35, 'Converged?', ha='left', va='center', 
               fontsize=9, fontweight='bold')
        
        # Add optimizer phase indicators
        phase1_box = FancyBboxPatch((0.05, 0.4), 0.35, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='lightblue', 
                                  edgecolor='blue', 
                                  linewidth=2, alpha=0.3)
        ax.add_patch(phase1_box)
        
        ax.text(0.225, 0.52, 'Phase 1: Adam Optimization', ha='center', va='center', 
               fontsize=11, fontweight='bold', color='blue')
        
        phase2_box = FancyBboxPatch((0.6, 0.4), 0.35, 0.15, 
                                  boxstyle="round,pad=0.02", 
                                  facecolor='lightgreen', 
                                  edgecolor='green', 
                                  linewidth=2, alpha=0.3)
        ax.add_patch(phase2_box)
        
        ax.text(0.775, 0.52, 'Phase 2: L-BFGS Refinement', ha='center', va='center', 
               fontsize=11, fontweight='bold', color='green')
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0.1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def create_data_flow_diagram(self, 
                               title: str = "PINN Data Flow",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create data flow diagram showing how data moves through the PINN system.
        
        Args:
            title: Diagram title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Define data flow components
        components = {
            'LAS Files': {
                'position': (0.1, 0.8),
                'size': (0.15, 0.08),
                'color': '#E8F4FD',
                'description': 'Raw well log data\n(.las files)'
            },
            'Data Preprocessing': {
                'position': (0.35, 0.8),
                'size': (0.15, 0.08),
                'color': self.colors['data'],
                'description': 'Cleaning, filtering,\nnormalization'
            },
            'Training Dataset': {
                'position': (0.6, 0.8),
                'size': (0.15, 0.08),
                'color': '#B3D9FF',
                'description': 'Processed training\ndata tensors'
            },
            'PINN Model': {
                'position': (0.5, 0.5),
                'size': (0.2, 0.1),
                'color': self.colors['output'],
                'description': 'Neural network with\nphysics constraints'
            },
            'Physics Engine': {
                'position': (0.2, 0.3),
                'size': (0.15, 0.08),
                'color': self.colors['physics'],
                'description': 'PDE residuals\ncomputation'
            },
            'Loss Function': {
                'position': (0.8, 0.3),
                'size': (0.15, 0.08),
                'color': self.colors['loss'],
                'description': 'Combined loss\ncomputation'
            },
            'Optimizer': {
                'position': (0.5, 0.2),
                'size': (0.15, 0.08),
                'color': '#3498DB',
                'description': 'Parameter updates\n(Adam/L-BFGS)'
            },
            'Predictions': {
                'position': (0.85, 0.6),
                'size': (0.12, 0.08),
                'color': '#1ABC9C',
                'description': 'Pressure &\nsaturation fields'
            }
        }
        
        # Draw components
        component_positions = {}
        for comp_name, details in components.items():
            x, y = details['position']
            w, h = details['size']
            
            # Create component box
            box = FancyBboxPatch((x - w/2, y - h/2), w, h, 
                               boxstyle="round,pad=0.01", 
                               facecolor=details['color'], 
                               edgecolor=self.colors['border'], 
                               linewidth=2, alpha=0.8)
            ax.add_patch(box)
            component_positions[comp_name] = (x, y)
            
            # Add component name
            ax.text(x, y + 0.02, comp_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white' if details['color'] != '#E8F4FD' else 'black')
            
            # Add description
            ax.text(x, y - 0.02, details['description'], ha='center', va='center', 
                   fontsize=8, color='white' if details['color'] != '#E8F4FD' else 'black', 
                   style='italic')
        
        # Define data flows
        flows = [
            ('LAS Files', 'Data Preprocessing', 'Raw data'),
            ('Data Preprocessing', 'Training Dataset', 'Clean data'),
            ('Training Dataset', 'PINN Model', 'Input features'),
            ('PINN Model', 'Physics Engine', 'Predictions'),
            ('PINN Model', 'Predictions', 'Output'),
            ('Physics Engine', 'Loss Function', 'PDE residuals'),
            ('Training Dataset', 'Loss Function', 'Target values'),
            ('Loss Function', 'Optimizer', 'Gradients'),
            ('Optimizer', 'PINN Model', 'Updated weights')
        ]
        
        # Draw data flows
        for start_comp, end_comp, flow_label in flows:
            if start_comp in component_positions and end_comp in component_positions:
                x1, y1 = component_positions[start_comp]
                x2, y2 = component_positions[end_comp]
                
                # Calculate arrow positions
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Offset from box edges
                    offset = 0.08
                    start_x = x1 + (dx / length) * offset
                    start_y = y1 + (dy / length) * offset
                    end_x = x2 - (dx / length) * offset
                    end_y = y2 - (dy / length) * offset
                    
                    # Draw arrow
                    arrow = patches.FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                                  arrowstyle='->', mutation_scale=15, 
                                                  color=self.colors['arrow'], linewidth=2)
                    ax.add_patch(arrow)
                    
                    # Add flow label
                    mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
                    ax.text(mid_x, mid_y, flow_label, ha='center', va='center', 
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='white', alpha=0.8))
        
        # Add feedback loop
        feedback_arrow = patches.FancyArrowPatch((0.5, 0.15), (0.5, 0.45),
                                               arrowstyle='->', mutation_scale=15, 
                                               color='red', linewidth=2, linestyle='--',
                                               connectionstyle="arc3,rad=0.3")
        ax.add_patch(feedback_arrow)
        
        ax.text(0.3, 0.1, 'Training Loop', ha='center', va='center', 
               fontsize=10, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add data types legend
        legend_elements = [
            patches.Patch(color='#E8F4FD', label='Raw Data'),
            patches.Patch(color=self.colors['data'], label='Processed Data'),
            patches.Patch(color=self.colors['output'], label='Model'),
            patches.Patch(color=self.colors['physics'], label='Physics'),
            patches.Patch(color=self.colors['loss'], label='Loss/Optimization'),
            patches.Patch(color='#1ABC9C', label='Results')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98))
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filepath: str, dpi: int = 300):
        """
        Save figure with consistent formatting.
        
        Args:
            fig: matplotlib Figure object
            filepath: Path to save the figure
            dpi: Resolution for saved figure
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)