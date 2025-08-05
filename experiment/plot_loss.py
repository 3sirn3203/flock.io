import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def extract_loss_from_logs(log_history: List[Dict]) -> tuple:
    """
    Extract train and validation losses from trainer log history
    
    Args:
        log_history: List of log dictionaries from trainer.state.log_history
        
    Returns:
        tuple: (train_steps, train_losses, eval_steps, eval_losses)
    """
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    
    for log in log_history:
        if 'loss' in log and 'step' in log:
            train_steps.append(log['step'])
            train_losses.append(log['loss'])
        
        if 'eval_loss' in log and 'step' in log:
            eval_steps.append(log['step'])
            eval_losses.append(log['eval_loss'])
    
    return train_steps, train_losses, eval_steps, eval_losses


def plot_training_losses(train_steps: List[int], train_losses: List[float],
                        eval_steps: List[int], eval_losses: List[float],
                        output_dir: str = "outputs", 
                        title: str = "Training and Validation Loss",
                        save_name: str = "loss_plot.png") -> None:
    """
    Plot training and validation losses
    
    Args:
        train_steps: List of training steps
        train_losses: List of training losses
        eval_steps: List of evaluation steps
        eval_losses: List of evaluation losses
        output_dir: Directory to save the plot
        title: Title of the plot
        save_name: Name of the saved plot file
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    if train_steps and train_losses:
        plt.plot(train_steps, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    
    # Plot validation loss
    if eval_steps and eval_losses:
        plt.plot(eval_steps, eval_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Show the plot
    plt.show()
    
    print(f"Loss plot saved to: {save_path}")


def plot_loss_from_trainer(trainer, output_dir: str = "outputs") -> None:
    """
    Extract losses from trainer and create plot
    
    Args:
        trainer: The trained SFTTrainer object
        output_dir: Directory to save the plot
    """
    if not hasattr(trainer, 'state') or not hasattr(trainer.state, 'log_history'):
        print("Warning: No log history found in trainer")
        return
    
    # Extract losses from log history
    train_steps, train_losses, eval_steps, eval_losses = extract_loss_from_logs(trainer.state.log_history)
    
    if not train_losses and not eval_losses:
        print("Warning: No loss data found in trainer logs")
        return
    
    # Create the plot
    plot_training_losses(
        train_steps, train_losses, 
        eval_steps, eval_losses,
        output_dir=output_dir,
        title="LoRA Fine-tuning: Training and Validation Loss"
    )
    
    # Save loss data as JSON for future analysis
    loss_data = {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses
    }
    
    json_path = os.path.join(output_dir, "loss_data.json")
    with open(json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    print(f"Loss data saved to: {json_path}")


def plot_loss_from_json(json_path: str, output_dir: str = "outputs") -> None:
    """
    Load loss data from JSON file and create plot
    
    Args:
        json_path: Path to the JSON file containing loss data
        output_dir: Directory to save the plot
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} not found")
        return
    
    with open(json_path, 'r') as f:
        loss_data = json.load(f)
    
    train_steps = loss_data.get("train_steps", [])
    train_losses = loss_data.get("train_losses", [])
    eval_steps = loss_data.get("eval_steps", [])
    eval_losses = loss_data.get("eval_losses", [])
    
    plot_training_losses(
        train_steps, train_losses,
        eval_steps, eval_losses,
        output_dir=output_dir,
        title="LoRA Fine-tuning: Training and Validation Loss (from saved data)"
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot training losses")
    parser.add_argument("--json_path", type=str, help="Path to loss data JSON file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    if args.json_path:
        plot_loss_from_json(args.json_path, args.output_dir)
    else:
        print("Please provide --json_path argument")