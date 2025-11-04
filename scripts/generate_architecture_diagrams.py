#!/usr/bin/env python3
"""
Generate comprehensive architecture diagrams for PINN Tutorial System.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_architecture_diagrams():
    """Create comprehensive PINN architecture diagrams."""
    
    output_dir = Path("architecture_diagrams")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating PINN architecture diagrams...")
    
    # 1. High-Level System Architecture
    create_system_architecture(output_dir)
    
    # 2. Data Processing Pipeline
    create_data_pipeline_diagram(output_dir)
    
    # 3. PINN Model Architecture
    create_model_architecture_diagram(output_dir)
    
    # 4. Training Process Flow
    create_training_flow_diagram(output_dir)
    
    print("✓ All architecture diagrams created successfully!")
    print(f"Diagrams saved to: {output_dir}")

def create_system_architecture(output_dir):
    """Create high-level system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define components with positions and colors
    components = [
        # Data Layer
        {'name': 'LAS File\nReader', 'pos': (1, 9), 'size': (1.5, 1), 'color': 'lightblue'},
        {'name': 'Data\nPreprocessor', 'pos': (3, 9), 'size': (1.5, 1), 'color': 'lightblue'},
        {'name': 'Dataset\nBuilder', 'pos': (5, 9), 'size': (1.5, 1), 'color': 'lightblue'},
        {'name': 'Quality\nAssessor', 'pos': (7, 9), 'size': (1.5, 1), 'color': 'lightblue'},
        
        # Model Layer
        {'name': 'PINN\nArchitecture', 'pos': (2, 7), 'size': (1.5, 1), 'color': 'lightgreen'},
        {'name': 'Physics\nEngine', 'pos': (4, 7), 'size': (1.5, 1), 'color': 'lightgreen'},
        {'name': 'Tensor\nManager', 'pos': (6, 7), 'size': (1.5, 1), 'color': 'lightgreen'},
        
        # Training Layer
        {'name': 'Optimizer\nManager', 'pos': (1, 5), 'size': (1.5, 1), 'color': 'orange'},
        {'name': 'Loss\nCalculator', 'pos': (3, 5), 'size': (1.5, 1), 'color': 'orange'},
        {'name': 'Convergence\nMonitor', 'pos': (5, 5), 'size': (1.5, 1), 'color': 'orange'},
        {'name': 'Batch\nProcessor', 'pos': (7, 5), 'size': (1.5, 1), 'color': 'orange'},
        
        # Validation Layer
        {'name': 'Cross\nValidator', 'pos': (2, 3), 'size': (1.5, 1), 'color': 'pink'},
        {'name': 'Physics\nChecker', 'pos': (4, 3), 'size': (1.5, 1), 'color': 'pink'},
        {'name': 'Performance\nBenchmark', 'pos': (6, 3), 'size': (1.5, 1), 'color': 'pink'},
        
        # Visualization Layer
        {'name': 'Scientific\nPlotter', 'pos': (1, 1), 'size': (1.5, 1), 'color': 'lightyellow'},
        {'name': 'Training\nVisualizer', 'pos': (3, 1), 'size': (1.5, 1), 'color': 'lightyellow'},
        {'name': 'Results\nAnalyzer', 'pos': (5, 1), 'size': (1.5, 1), 'color': 'lightyellow'},
        {'name': 'Diagram\nGenerator', 'pos': (7, 1), 'size': (1.5, 1), 'color': 'lightyellow'},
    ]
    
    # Draw components
    for comp in components:
        # Component box
        rect = FancyBboxPatch(
            comp['pos'], comp['size'][0], comp['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Component text
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2,
               comp['name'], ha='center', va='center', fontsize=10, weight='bold')
    
    # Add title
    ax.set_title('PINN Tutorial System - High-Level Architecture', fontsize=18, weight='bold', pad=20)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created system architecture diagram")

def create_data_pipeline_diagram(output_dir):
    """Create detailed data processing pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # Pipeline stages
    stages = [
        {'name': 'Raw LAS Files\n(767 KGS Wells)', 'pos': (1, 8), 'color': 'lightcoral'},
        {'name': 'LAS Parser\n& Reader', 'pos': (1, 6.5), 'color': 'lightblue'},
        {'name': 'Quality Control\n& Validation', 'pos': (1, 5), 'color': 'lightgreen'},
        {'name': 'Data Cleaning\n& Preprocessing', 'pos': (1, 3.5), 'color': 'orange'},
        {'name': 'Feature Engineering\n& Derivation', 'pos': (1, 2), 'color': 'yellow'},
        {'name': 'Dataset Builder\n& Splitting', 'pos': (1, 0.5), 'color': 'lightpink'},
    ]
    
    # Results for each stage
    results = [
        '767 files\n681,131 points',
        '92 parsed\n(12% success)',
        '30 quality\n(67% of parsed)',
        '137,900 clean\ndata points',
        '20 wells with\nderived properties',
        'Train: 89,346\nVal: 24,621\nTest: 23,933'
    ]
    
    # Draw pipeline stages
    for i, (stage, result) in enumerate(zip(stages, results)):
        y_pos = stage['pos'][1]
        
        # Main stage box
        stage_rect = FancyBboxPatch(
            (0.5, y_pos-0.3), 2, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(stage_rect)
        ax.text(1.5, y_pos, stage['name'], ha='center', va='center', fontsize=11, weight='bold')
        
        # Results box
        result_rect = FancyBboxPatch(
            (9.5, y_pos-0.3), 2, 0.6,
            boxstyle="round,pad=0.1",
            facecolor='lightsteelblue',
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(result_rect)
        ax.text(10.5, y_pos, result, ha='center', va='center', fontsize=10, weight='bold')
        
        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate('', xy=(1.5, y_pos-0.7), xytext=(1.5, y_pos-0.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_title('PINN Data Processing Pipeline - Detailed Flow', fontsize=18, weight='bold', pad=20)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 9.5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_data_pipeline_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created data pipeline diagram")

def create_model_architecture_diagram(output_dir):
    """Create detailed PINN model architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Input layer
    input_nodes = ['Depth', 'Gamma Ray', 'Porosity', 'Permeability']
    input_y = [8, 7, 6, 5]
    
    for i, (node, y) in enumerate(zip(input_nodes, input_y)):
        circle = plt.Circle((1, y), 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(1, y, f'x_{i+1}', ha='center', va='center', fontsize=10, weight='bold')
        ax.text(0.2, y, node, ha='right', va='center', fontsize=10)
    
    # Hidden layers
    hidden_layers = [
        {'x': 3, 'nodes': 8, 'label': 'Hidden Layer 1\n(64 neurons)'},
        {'x': 5, 'nodes': 8, 'label': 'Hidden Layer 2\n(64 neurons)'},
        {'x': 7, 'nodes': 8, 'label': 'Hidden Layer 3\n(64 neurons)'}
    ]
    
    for layer in hidden_layers:
        y_positions = np.linspace(8.5, 4.5, layer['nodes'])
        for y in y_positions:
            circle = plt.Circle((layer['x'], y), 0.2, color='lightgreen', ec='black', linewidth=1)
            ax.add_patch(circle)
        
        # Layer label
        ax.text(layer['x'], 3.5, layer['label'], ha='center', va='center', 
               fontsize=10, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Output layer
    output_nodes = ['Pressure', 'Saturation']
    output_y = [7, 5.5]
    output_colors = ['orange', 'pink']
    
    for i, (node, y, color) in enumerate(zip(output_nodes, output_y, output_colors)):
        circle = plt.Circle((9, y), 0.3, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(9, y, f'u_{i+1}', ha='center', va='center', fontsize=10, weight='bold')
        ax.text(9.8, y, node, ha='left', va='center', fontsize=10)
    
    # Physics constraints
    physics_rect = FancyBboxPatch(
        (4, 1), 3, 1,
        boxstyle="round,pad=0.1",
        facecolor='lightyellow',
        edgecolor='red',
        linewidth=2
    )
    ax.add_patch(physics_rect)
    ax.text(5.5, 1.5, 'Physics Constraints\n∇·(k∇p) = 0\nS_o + S_w = 1', 
           ha='center', va='center', fontsize=10, weight='bold')
    
    ax.set_title('PINN Model Architecture - Neural Network + Physics Integration', 
                fontsize=18, weight='bold', pad=20)
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_model_architecture_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created model architecture diagram")

def create_training_flow_diagram(output_dir):
    """Create training process flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    
    # Training phases
    phases = [
        {
            'name': 'Phase 1: Data Fitting',
            'epochs': 'Epochs 1-200',
            'pos': (2, 12),
            'color': 'lightblue'
        },
        {
            'name': 'Phase 2: Physics Integration',
            'epochs': 'Epochs 201-800',
            'pos': (9, 12),
            'color': 'lightgreen'
        },
        {
            'name': 'Phase 3: Fine-tuning',
            'epochs': 'Epochs 801-1252',
            'pos': (16, 12),
            'color': 'orange'
        }
    ]
    
    # Draw phases
    for phase in phases:
        # Phase box
        rect = FancyBboxPatch(
            (phase['pos'][0]-1.5, phase['pos'][1]-1), 3, 2,
            boxstyle="round,pad=0.1",
            facecolor=phase['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Phase title
        ax.text(phase['pos'][0], phase['pos'][1]+0.5, phase['name'], 
               ha='center', va='center', fontsize=12, weight='bold')
        ax.text(phase['pos'][0], phase['pos'][1]-0.5, phase['epochs'], 
               ha='center', va='center', fontsize=10, style='italic')
    
    # Training loop components
    loop_components = [
        {'name': 'Batch Loading', 'pos': (2, 8), 'color': 'lightcoral'},
        {'name': 'Forward Pass', 'pos': (5, 8), 'color': 'lightblue'},
        {'name': 'Loss Calculation', 'pos': (8, 8), 'color': 'lightgreen'},
        {'name': 'Backward Pass', 'pos': (11, 8), 'color': 'orange'},
        {'name': 'Parameter Update', 'pos': (14, 8), 'color': 'pink'},
        {'name': 'Validation Check', 'pos': (17, 8), 'color': 'lightyellow'}
    ]
    
    for i, comp in enumerate(loop_components):
        # Component box
        rect = FancyBboxPatch(
            (comp['pos'][0]-0.8, comp['pos'][1]-0.4), 1.6, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'], 
               ha='center', va='center', fontsize=10, weight='bold')
        
        # Arrow to next component
        if i < len(loop_components) - 1:
            ax.annotate('', xy=(loop_components[i+1]['pos'][0]-0.8, comp['pos'][1]), 
                       xytext=(comp['pos'][0]+0.8, comp['pos'][1]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
    
    ax.set_title('PINN Training Process - Multi-Phase Strategy', 
                fontsize=18, weight='bold', pad=20)
    ax.set_xlim(0, 19)
    ax.set_ylim(6, 14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_training_process_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created training flow diagram")

if __name__ == "__main__":
    create_architecture_diagrams()