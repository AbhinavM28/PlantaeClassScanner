"""
Generate comprehensive metadata for PlantaeClassScanner dataset
Validates images, checks for duplicates, and creates statistics

Author: Abhinav M
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.logger import setup_logger
    logger = setup_logger("MetadataGenerator")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MetadataGenerator")

try:
    from PIL import Image
except ImportError:
    logger.error("Pillow not installed. Install with: pip install pillow")
    sys.exit(1)


class MetadataGenerator:
    """Generate and validate dataset metadata"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.metadata = {
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_images": 0,
            "classes": {},
            "image_stats": {
                "formats": defaultdict(int),
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "mean_width": 0,
                "mean_height": 0,
                "min_width": float('inf'),
                "max_width": 0,
                "min_height": float('inf'),
                "max_height": 0
            },
            "splits": {
                "train": {"count": 0, "percentage": 0},
                "val": {"count": 0, "percentage": 0},
                "test": {"count": 0, "percentage": 0}
            },
            "quality_checks": {
                "corrupt_images": [],
                "duplicate_images": [],
                "low_resolution": [],
                "aspect_ratio_outliers": []
            }
        }
        self.image_hashes = {}  # hash -> [paths]
        self.image_sizes = []
    
    def compute_file_hash(self, file_path):
        """
        Compute MD5 hash of file for duplicate detection
        
        Args:
            file_path: Path to file
            
        Returns:
            str: MD5 hash
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def analyze_image(self, img_path, class_name, split_name=None):
        """
        Analyze a single image and update statistics
        
        Args:
            img_path: Path to image
            class_name: Class the image belongs to
            split_name: Split name (train/val/test) or None
        """
        try:
            # Open and verify image
            with Image.open(img_path) as img:
                width, height = img.size
                img_format = img.format
                
                # Verify image can be loaded
                img.verify()
            
            # Get file size
            file_size = os.path.getsize(img_path)
            
            # Compute hash for duplicate detection
            img_hash = self.compute_file_hash(img_path)
            
            # Check for duplicates
            if img_hash in self.image_hashes:
                self.image_hashes[img_hash].append(str(img_path))
                self.metadata["quality_checks"]["duplicate_images"].append({
                    "hash": img_hash,
                    "paths": self.image_hashes[img_hash]
                })
            else:
                self.image_hashes[img_hash] = [str(img_path)]
            
            # Check resolution
            if width < 224 or height < 224:
                self.metadata["quality_checks"]["low_resolution"].append({
                    "path": str(img_path),
                    "size": f"{width}x{height}"
                })
            
            # Check aspect ratio (flag extreme outliers)
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                self.metadata["quality_checks"]["aspect_ratio_outliers"].append({
                    "path": str(img_path),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "size": f"{width}x{height}"
                })
            
            # Update statistics
            self.metadata["total_images"] += 1
            self.metadata["classes"][class_name]["count"] += 1
            self.metadata["image_stats"]["formats"][img_format] += 1
            self.metadata["image_stats"]["total_size_bytes"] += file_size
            self.image_sizes.append((width, height))
            
            # Update min/max dimensions
            self.metadata["image_stats"]["min_width"] = min(
                self.metadata["image_stats"]["min_width"], width
            )
            self.metadata["image_stats"]["max_width"] = max(
                self.metadata["image_stats"]["max_width"], width
            )
            self.metadata["image_stats"]["min_height"] = min(
                self.metadata["image_stats"]["min_height"], height
            )
            self.metadata["image_stats"]["max_height"] = max(
                self.metadata["image_stats"]["max_height"], height
            )
            
            # Update split counts
            if split_name:
                self.metadata["classes"][class_name][split_name] += 1
                self.metadata["splits"][split_name]["count"] += 1
            
            # Store example paths (first 3 per class)
            if len(self.metadata["classes"][class_name]["examples"]) < 3:
                self.metadata["classes"][class_name]["examples"].append(str(img_path))
        
        except Exception as e:
            self.metadata["quality_checks"]["corrupt_images"].append({
                "path": str(img_path),
                "error": str(e)
            })
            logger.debug(f"Corrupt image: {img_path} - {e}")
    
    def analyze_directory(self, dir_path, split_name=None):
        """
        Analyze all images in a directory
        
        Args:
            dir_path: Path to directory
            split_name: Name of split (train/val/test) or None
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return
        
        logger.info(f"📂 Analyzing: {dir_path}")
        
        # Iterate through class directories
        for class_dir in sorted(dir_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Initialize class entry if needed
            if class_name not in self.metadata["classes"]:
                self.metadata["classes"][class_name] = {
                    "count": 0,
                    "train": 0,
                    "val": 0,
                    "test": 0,
                    "source": "iNaturalist",
                    "verified": True,
                    "examples": []
                }
            
            # Find all images
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(class_dir.glob(ext))
            
            logger.info(f"  {class_name}: {len(image_files)} images")
            
            # Analyze each image
            for img_path in image_files:
                self.analyze_image(img_path, class_name, split_name)
    
    def calculate_statistics(self):
        """Calculate aggregate statistics"""
        # Calculate mean dimensions
        if self.image_sizes:
            widths = [s[0] for s in self.image_sizes]
            heights = [s[1] for s in self.image_sizes]
            
            self.metadata["image_stats"]["mean_width"] = int(sum(widths) / len(widths))
            self.metadata["image_stats"]["mean_height"] = int(sum(heights) / len(heights))
        
        # Convert total size to MB
        self.metadata["image_stats"]["total_size_mb"] = round(
            self.metadata["image_stats"]["total_size_bytes"] / (1024 * 1024), 2
        )
        
        # Convert formats dict
        self.metadata["image_stats"]["formats"] = dict(
            self.metadata["image_stats"]["formats"]
        )
        
        # Calculate split percentages
        if self.metadata["total_images"] > 0:
            for split_name in ["train", "val", "test"]:
                count = self.metadata["splits"][split_name]["count"]
                percentage = (count / self.metadata["total_images"]) * 100
                self.metadata["splits"][split_name]["percentage"] = round(percentage, 1)
        
        # Calculate class balance ratio
        if self.metadata["classes"]:
            class_counts = [c["count"] for c in self.metadata["classes"].values()]
            if class_counts:
                self.metadata["balance_ratio"] = round(
                    min(class_counts) / max(class_counts), 3
                )
    
    def generate(self):
        """Generate complete metadata"""
        logger.info("=" * 70)
        logger.info("🔍 Generating Dataset Metadata")
        logger.info("=" * 70)
        logger.info(f"Data directory: {self.data_dir.absolute()}")
        logger.info("")
        
        # Analyze each split
        for split_name in ["train", "val", "test"]:
            split_dir = self.data_dir / split_name
            if split_dir.exists():
                self.analyze_directory(split_dir, split_name)
        
        # Also analyze raw directory if present
        raw_dir = self.data_dir / "raw"
        if raw_dir.exists():
            logger.info("\n📂 Analyzing raw directory (not included in splits)...")
            # Don't analyze raw if splits exist (avoid double counting)
            if not (self.data_dir / "train").exists():
                self.analyze_directory(raw_dir)
        
        # Calculate aggregate stats
        logger.info("\nCalculating statistics...")
        self.calculate_statistics()
        
        # Save metadata
        output_path = self.data_dir / "dataset_metadata.json"
        with open(output_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"\n✅ Metadata saved to: {output_path}")
        
        # Print summary
        self.print_summary()
        
        return output_path
    
    def print_summary(self):
        """Print human-readable summary"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 DATASET SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Images: {self.metadata['total_images']}")
        logger.info(f"Number of Classes: {len(self.metadata['classes'])}")
        logger.info(f"Total Size: {self.metadata['image_stats']['total_size_mb']} MB")
        logger.info(f"Balance Ratio: {self.metadata.get('balance_ratio', 0):.1%}")
        
        logger.info(f"\n📏 Image Dimensions:")
        logger.info(f"  Mean: {self.metadata['image_stats']['mean_width']}x{self.metadata['image_stats']['mean_height']}")
        logger.info(f"  Min:  {self.metadata['image_stats']['min_width']}x{self.metadata['image_stats']['min_height']}")
        logger.info(f"  Max:  {self.metadata['image_stats']['max_width']}x{self.metadata['image_stats']['max_height']}")
        
        logger.info(f"\n📊 Class Distribution:")
        for class_name, stats in sorted(self.metadata["classes"].items()):
            logger.info(f"  {class_name:30s} {stats['count']:4d} images")
        
        logger.info(f"\n🔀 Split Distribution:")
        for split, stats in self.metadata["splits"].items():
            if stats["count"] > 0:
                logger.info(f"  {split:5s} {stats['count']:4d} images ({stats['percentage']:5.1f}%)")
        
        logger.info(f"\n⚠️  Quality Issues:")
        logger.info(f"  Corrupt Images: {len(self.metadata['quality_checks']['corrupt_images'])}")
        logger.info(f"  Duplicate Images: {len(self.metadata['quality_checks']['duplicate_images'])}")
        logger.info(f"  Low Resolution (<224px): {len(self.metadata['quality_checks']['low_resolution'])}")
        logger.info(f"  Aspect Ratio Outliers: {len(self.metadata['quality_checks']['aspect_ratio_outliers'])}")
        
        # Show warnings if issues found
        if self.metadata['quality_checks']['corrupt_images']:
            logger.warning(f"\n⚠️  Found {len(self.metadata['quality_checks']['corrupt_images'])} corrupt images!")
        
        if self.metadata['quality_checks']['low_resolution']:
            logger.warning(f"\n⚠️  Found {len(self.metadata['quality_checks']['low_resolution'])} low resolution images!")
        
        if self.metadata.get('balance_ratio', 1.0) < 0.8:
            logger.warning(f"\n⚠️  Dataset is imbalanced (ratio: {self.metadata.get('balance_ratio', 0):.1%})")
            logger.warning("     Consider collecting more images for underrepresented classes")
        
        logger.info("=" * 70)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate dataset metadata")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)"
    )
    
    args = parser.parse_args()
    
    generator = MetadataGenerator(data_dir=args.data_dir)
    generator.generate()


if __name__ == "__main__":
    main()