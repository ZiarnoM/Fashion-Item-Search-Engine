import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from torchvision import transforms
from stanford_products_loader import StanfordProductsDataset, create_stanford_loaders
import seaborn as sns


def analyze_split(dataset, split_name):
    """Analyze a single split"""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} SPLIT ANALYSIS")
    print(f"{'='*70}")
    
    # Get labels and categories
    labels = dataset.labels  # Product IDs
    categories = dataset.super_labels  # Categories (1-12)
    
    # Count products
    product_counts = Counter(labels)
    category_counts = Counter(categories)
    
    print(f"\nTotal samples: {len(labels):,}")
    print(f"Unique products: {len(product_counts):,}")
    print(f"Unique categories: {len(category_counts)}")
    
    # Product-level statistics
    samples_per_product = list(product_counts.values())
    print(f"\n--- PRODUCT-LEVEL STATS (Critical for Metric Learning!) ---")
    print(f"Min samples per product: {min(samples_per_product)}")
    print(f"Max samples per product: {max(samples_per_product)}")
    print(f"Mean samples per product: {np.mean(samples_per_product):.2f}")
    print(f"Median samples per product: {np.median(samples_per_product):.1f}")
    
    # WARNING: Products with only 1 sample
    single_sample_products = sum(1 for count in samples_per_product if count == 1)
    print(f"\n️  Products with only 1 sample: {single_sample_products} ({single_sample_products/len(product_counts)*100:.1f}%)")
    
    if single_sample_products > len(product_counts) * 0.3:
        print("    PROBLEM: >30% of products have only 1 sample!")
        print("   This makes metric learning impossible (no positive pairs)")
    
    # Products with 2 samples
    two_sample_products = sum(1 for count in samples_per_product if count == 2)
    print(f"   Products with 2 samples: {two_sample_products} ({two_sample_products/len(product_counts)*100:.1f}%)")
    
    # Category-level statistics
    print(f"\n--- CATEGORY-LEVEL STATS ---")
    for cat_id in sorted(category_counts.keys()):
        count = category_counts[cat_id]
        print(f"Category {cat_id:2d}: {count:5,} samples")
    
    return {
        'split_name': split_name,
        'total_samples': len(labels),
        'num_products': len(product_counts),
        'num_categories': len(category_counts),
        'product_counts': product_counts,
        'category_counts': category_counts,
        'samples_per_product': samples_per_product,
        'single_sample_products': single_sample_products
    }


def visualize_distribution(train_stats, val_stats, test_stats):
    """Create visualization of class distributions"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Samples per product distribution
    for idx, (stats, color) in enumerate([(train_stats, 'blue'), 
                                            (val_stats, 'orange'), 
                                            (test_stats, 'green')]):
        ax = axes[0, idx]
        
        samples = stats['samples_per_product']
        bins = range(1, min(max(samples) + 2, 50))  # Cap at 50 for readability
        
        ax.hist(samples, bins=bins, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Samples per Product', fontsize=11)
        ax.set_ylabel('Number of Products', fontsize=11)
        ax.set_title(f'{stats["split_name"]} - Samples per Product', fontsize=12, fontweight='bold')
        ax.axvline(np.mean(samples), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(samples):.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text with critical info
        single = stats['single_sample_products']
        total = stats['num_products']
        ax.text(0.98, 0.98, f'Single-sample products:\n{single}/{total} ({single/total*100:.1f}%)',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Category distribution
    for idx, (stats, color) in enumerate([(train_stats, 'blue'), 
                                            (val_stats, 'orange'), 
                                            (test_stats, 'green')]):
        ax = axes[1, idx]
        
        cat_counts = stats['category_counts']
        categories = sorted(cat_counts.keys())
        counts = [cat_counts[c] for c in categories]
        
        ax.bar(categories, counts, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Category ID', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title(f'{stats["split_name"]} - Samples per Category', fontsize=12, fontweight='bold')
        ax.set_xticks(categories)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (cat, count) in enumerate(zip(categories, counts)):
            ax.text(cat, count, f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/dataset_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to results/dataset_distribution_analysis.png")
    plt.close()


def check_validation_viability(val_stats):
    """Check if validation set is viable for metric learning"""
    print(f"\n{'='*70}")
    print("VALIDATION SET VIABILITY CHECK")
    print(f"{'='*70}")
    
    single_sample = val_stats['single_sample_products']
    total_products = val_stats['num_products']
    single_pct = single_sample / total_products * 100
    
    samples_per_product = val_stats['samples_per_product']
    avg_samples = np.mean(samples_per_product)
    
    print(f"\nValidation set has:")
    print(f"  Total products: {total_products}")
    print(f"  Products with 1 sample: {single_sample} ({single_pct:.1f}%)")
    print(f"  Average samples per product: {avg_samples:.2f}")
    
    # Diagnosis
    print(f"\n--- DIAGNOSIS ---")
    
    if single_pct > 50:
        print(" CRITICAL: >50% of products have only 1 sample")
        print("   Problem: No positive pairs → model can't learn similarities")
        print("   Solution: Change train/val split strategy")
        return False
    elif single_pct > 30:
        print("️  WARNING: 30-50% of products have only 1 sample")
        print("   Problem: Limited positive pairs → high validation loss expected")
        print("   Note: This explains why val_loss >> train_loss")
        return True
    elif avg_samples < 2.0:
        print("  WARNING: Average <2 samples per product")
        print("   Problem: Very few positive pairs")
        return True
    else:
        print(" GOOD: Validation set is viable for metric learning")
        print(f"   Most products have multiple samples ({avg_samples:.1f} avg)")
        return True


def suggest_improvements(train_stats, val_stats):
    """Suggest improvements based on analysis"""
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    single_pct_val = val_stats['single_sample_products'] / val_stats['num_products'] * 100
    single_pct_train = train_stats['single_sample_products'] / train_stats['num_products'] * 100
    
    if single_pct_val > 30:
        print("\n1. Change Validation Split Strategy:")
        print("   Current: Random 10% of samples → many products get only 1 sample")
        print("   Better: Ensure each product has at least 2 samples in validation")
        print("   ")
        print("   Implementation: Use product-aware splitting")
        print("   This will reduce val_loss from ~1.4 to ~0.3-0.5")
    
    if single_pct_train > 10:
        print("\n2. Filter Training Data:")
        print("   Remove products with <3 samples from training")
        print("   This focuses model on products with enough data")
    
    print("\n3. Use Product-Based Split (NOT Sample-Based):")
    print("   Split products into train/val (not individual samples)")
    print("   Ensures all samples of a product stay together")


def main():
    print("Stanford Online Products - Dataset Distribution Analysis")
    print("="*70)
    
    # Load dataset with transforms
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load train dataset (contains train + val)
    print("\nLoading datasets...")
    train_dataset = StanfordProductsDataset(
        root_dir='./data/Stanford_Online_Products',
        split='train',
        transform=test_transform
    )
    
    # Get train split
    train_dataset.set_split('train')
    train_stats = analyze_split(train_dataset, 'Train')
    
    # Get val split
    train_dataset.set_split('val')
    val_stats = analyze_split(train_dataset, 'Validation')
    
    # Load test dataset
    test_dataset = StanfordProductsDataset(
        root_dir='./data/Stanford_Online_Products',
        split='test',
        transform=test_transform
    )
    test_stats = analyze_split(test_dataset, 'Test')
    
    # Visualize
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    visualize_distribution(train_stats, val_stats, test_stats)
    
    # Check validation viability
    is_viable = check_validation_viability(val_stats)
    
    # Suggest improvements
    if not is_viable or val_stats['single_sample_products'] / val_stats['num_products'] > 0.3:
        suggest_improvements(train_stats, val_stats)
    
    # Save report
    report = {
        'train': {
            'total_samples': train_stats['total_samples'],
            'num_products': train_stats['num_products'],
            'avg_samples_per_product': float(np.mean(train_stats['samples_per_product'])),
            'single_sample_products': train_stats['single_sample_products'],
            'single_sample_percentage': train_stats['single_sample_products'] / train_stats['num_products'] * 100
        },
        'validation': {
            'total_samples': val_stats['total_samples'],
            'num_products': val_stats['num_products'],
            'avg_samples_per_product': float(np.mean(val_stats['samples_per_product'])),
            'single_sample_products': val_stats['single_sample_products'],
            'single_sample_percentage': val_stats['single_sample_products'] / val_stats['num_products'] * 100,
            'is_viable': is_viable
        },
        'test': {
            'total_samples': test_stats['total_samples'],
            'num_products': test_stats['num_products'],
            'avg_samples_per_product': float(np.mean(test_stats['samples_per_product'])),
        }
    }
    
    with open('results/dataset_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved detailed report to results/dataset_analysis_report.json")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()