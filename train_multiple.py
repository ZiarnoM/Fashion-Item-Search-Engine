#!/usr/bin/env python3
"""
Script to train and evaluate multiple models sequentially
Usage: python train_multiple.py
"""

import subprocess
import time
import os
from datetime import datetime

# Configuration
CONFIG = {
    'dataset': 'stanford',  # or 'fashionmnist'
    'batch_size': 64,
    'epochs': 10,
    'embedding_size': 128,
    'margin': 0.5,
    'lr': 0.0001,
    'backbones': ['resnet18', 'resnet50', 'efficientnet_b0'],
    'eval_chunk_size': 1000
}


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"\n✓ {description} completed in {duration/60:.1f} minutes")
        return True, duration
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False, duration


def train_and_evaluate(config):
    """Train and evaluate multiple models"""
    
    print("="*60)
    print("TRAINING MULTIPLE MODELS SEQUENTIALLY")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Embedding size: {config['embedding_size']}")
    print(f"  Backbones: {', '.join(config['backbones'])}")
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_start = time.time()
    results = []
    
    for backbone in config['backbones']:
        model_start = time.time()
        
        print(f"\n{'#'*60}")
        print(f"# MODEL: {backbone}")
        print(f"{'#'*60}\n")
        
        # Training command
        train_cmd = [
            'python', 'train.py',
            '--dataset', config['dataset'],
            '--backbone', backbone,
            '--batch_size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--embedding_size', str(config['embedding_size']),
            '--margin', str(config['margin']),
            '--lr', str(config['lr'])
        ]
        
        train_success, train_duration = run_command(
            train_cmd,
            f"Training {backbone}"
        )
        
        eval_success = False
        eval_duration = 0
        
        if train_success:
            # Evaluation command
            model_path = f"checkpoints/{config['dataset']}_{backbone}_best.pth"
            
            if os.path.exists(model_path):
                eval_cmd = [
                    'python', 'evaluate.py',
                    '--model_path', model_path,
                    '--dataset', config['dataset'],
                    '--batch_size', str(config['batch_size']),
                    '--chunk_size', str(config['eval_chunk_size'])
                ]
                
                eval_success, eval_duration = run_command(
                    eval_cmd,
                    f"Evaluating {backbone}"
                )
            else:
                print(f"✗ Model file not found: {model_path}")
        else:
            print(f"✗ Skipping evaluation for {backbone} due to training failure")
        
        model_duration = time.time() - model_start
        
        # Store results
        results.append({
            'backbone': backbone,
            'train_success': train_success,
            'eval_success': eval_success,
            'train_time': train_duration,
            'eval_time': eval_duration,
            'total_time': model_duration
        })
        
        print(f"\nTime for {backbone}: {model_duration/60:.1f} minutes")
        
        # Clear GPU cache
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        
        # Short pause between models
        time.sleep(5)
    
    # Summary
    total_duration = time.time() - total_start
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETED!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration/3600:.1f} hours ({total_duration/60:.1f} minutes)")
    print("\nResults Summary:")
    print("-"*60)
    
    for r in results:
        status_train = "✓" if r['train_success'] else "✗"
        status_eval = "✓" if r['eval_success'] else "✗"
        print(f"{r['backbone']:20s} | Train: {status_train} ({r['train_time']/60:.1f}m) | "
              f"Eval: {status_eval} ({r['eval_time']/60:.1f}m) | "
              f"Total: {r['total_time']/60:.1f}m")
    
    print("\nModel files:")
    for backbone in config['backbones']:
        model_path = f"checkpoints/{config['dataset']}_{backbone}_best.pth"
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024*1024)
            print(f"  ✓ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {model_path} (not found)")
    
    print("\nResults saved in:")
    print(f"  - ./results/")
    print(f"  - ./checkpoints/")
    print("="*60)


if __name__ == '__main__':
    # Optional: Check if required files exist
    required_files = ['train.py', 'evaluate.py', 'stanford_products_loader.py']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"Error: Missing required files: {', '.join(missing)}")
        exit(1)
    
    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run training
    try:
        train_and_evaluate(CONFIG)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        exit(1)