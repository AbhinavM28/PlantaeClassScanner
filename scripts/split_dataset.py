"""
Split raw dataset into train/val/test sets with stratification
Maintains class balance across splits

Author: Abhinav M
Dependencies: split-folders
"""

import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.logger import setup_logger
    from src.utils.config_loader import get_config
    logger = setup_logger("DatasetSplitter")
    config = get_config()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DatasetSplitter")
    config = None

try:
    import splitfolders
except ImportError:
    logger.error("split-folders not installed. Install with: pip install split-folders")
    sys.exit(1)


def validate_raw_dataset(raw_dir):
    """
    Validate that raw dataset exists and has images
    
    Args:
        raw_dir: Path to raw dataset directory
        
    Returns:
        bool: True if valid, False otherwise
    """
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        logger.error(f"Raw dataset directory not found: {raw_path}")
        return False
    
    # Check for subdirectories (classes)
    class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        logger.error(f"No class directories found in {raw_path}")
        return False
    
    # Count total images
    total_images = 0
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        total_images += len(images)
        logger.info(f"  {class_dir.name}: {len(images)} images")
    
    if total_images == 0:
        logger.error("No images found in dataset!")
        return False
    
    logger.info(f"Total images found: {total_images}")
    return True


def clean_existing_splits(data_dir):
    """
    Clean existing train/val/test directories
    
    Args:
        data_dir: Parent data directory
    """
    data_path = Path(data_dir)
    
    for split_name in ["train", "val", "test"]:
        split_dir = data_path / split_name
        if split_dir.exists():
            logger.warning(f"Removing existing {split_name} directory...")
            shutil.rmtree(split_dir)


def split_dataset(raw_dir="data/raw", output_dir="data", 
                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        raw_dir: Directory containing raw images organized by class
        output_dir: Parent directory for train/val/test splits
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    logger.info("=" * 70)
    logger.info("📊 Dataset Splitting")
    logger.info("=" * 70)
    logger.info(f"Input directory: {raw_path.absolute()}")
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Split ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")
    logger.info(f"Random seed: {seed}")
    logger.info("")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(f"Split ratios must sum to 1.0 (current sum: {total_ratio})")
        return False
    
    # Validate raw dataset
    logger.info("Validating raw dataset...")
    if not validate_raw_dataset(raw_path):
        return False
    
    # Clean existing splits
    clean_existing_splits(output_path)
    
    # Perform split
    logger.info("\nSplitting dataset...")
    try:
        splitfolders.ratio(
            input=str(raw_path),
            output=str(output_path),
            seed=seed,
            ratio=(train_ratio, val_ratio, test_ratio),
            group_prefix=None,
            move=False  # Copy files, don't move
        )
        
        logger.info("\n✅ Dataset split complete!")
        logger.info(f"  📁 Train: {output_path / 'train'}")
        logger.info(f"  📁 Val:   {output_path / 'val'}")
        logger.info(f"  📁 Test:  {output_path / 'test'}")
        
        # Verify splits
        logger.info("\nVerifying splits...")
        for split_name in ["train", "val", "test"]:
            split_dir = output_path / split_name
            if split_dir.exists():
                total = sum(1 for _ in split_dir.rglob("*.jpg"))
                total += sum(1 for _ in split_dir.rglob("*.jpeg"))
                total += sum(1 for _ in split_dir.rglob("*.png"))
                logger.info(f"  {split_name}: {total} images")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        return False


def main():
    """Main execution"""
    # Get config values if available
    if config:
        data_dir = config.get('paths.data_dir', 'data')
        train_ratio = config.get('dataset.train_split', 0.70)
        val_ratio = config.get('dataset.val_split', 0.15)
        test_ratio = config.get('dataset.test_split', 0.15)
    else:
        data_dir = 'data'
        train_ratio = 0.70
        val_ratio = 0.15
        test_ratio = 0.15
    
    raw_dir = f"{data_dir}/raw"
    
    success = split_dataset(
        raw_dir=raw_dir,
        output_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()