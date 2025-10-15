"""
Generate dataset metadata for PlantaeClassScanner
Author: Abhinav M
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from PIL import Image
import hashlib

class MetadataGenerator:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.metadata = {
            "version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "total_images": 0,
            "classes": {},
            "image_stats": {
                "formats": defaultdict(int),
                "sizes": [],
                "total_size_mb": 0
            },
            "splits": {
                "train": {"count": 0, "percentage": 70},
                "val": {"count": 0, "percentage": 15},
                "test": {"count": 0, "percentage": 15}
            },
            "data_sources": [],
            "quality_checks": {
                "corrupt_images": [],
                "duplicate_images": [],
                "low_resolution": []
            }
        }
        self.image_hashes = set()
    
    def analyze_directory(self, dir_path, split_name=None):
        """Analyze a directory of images"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return
        
        print(f"\n📂 Analyzing: {dir_path}")
        
        for class_dir in dir_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.metadata["classes"]:
                self.metadata["classes"][class_name] = {
                    "count": 0,
                    "train": 0,
                    "val": 0,
                    "test": 0,
                    "source": "unknown",
                    "verified": False,
                    "examples": []
                }
            
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            
            for img_path in image_files:
                try:
                    # Basic checks
                    img = Image.open(img_path)
                    width, height = img.size
                    file_size = os.path.getsize(img_path)
                    
                    # Check for duplicates (via hash)
                    with open(img_path, 'rb') as f:
                        img_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if img_hash in self.image_hashes:
                        self.metadata["quality_checks"]["duplicate_images"].append(str(img_path))
                        continue
                    
                    self.image_hashes.add(img_hash)
                    
                    # Check resolution
                    if width < 224 or height < 224:
                        self.metadata["quality_checks"]["low_resolution"].append({
                            "path": str(img_path),
                            "size": f"{width}x{height}"
                        })
                    
                    # Update stats
                    self.metadata["total_images"] += 1
                    self.metadata["classes"][class_name]["count"] += 1
                    self.metadata["image_stats"]["formats"][img.format] += 1
                    self.metadata["image_stats"]["sizes"].append((width, height))
                    self.metadata["image_stats"]["total_size_mb"] += file_size / (1024 * 1024)
                    
                    if split_name:
                        self.metadata["classes"][class_name][split_name] += 1
                        self.metadata["splits"][split_name]["count"] += 1
                    
                    # Store example paths (first 3 per class)
                    if len(self.metadata["classes"][class_name]["examples"]) < 3:
                        self.metadata["classes"][class_name]["examples"].append(str(img_path))
                    
                    img.close()
                    
                except Exception as e:
                    self.metadata["quality_checks"]["corrupt_images"].append({
                        "path": str(img_path),
                        "error": str(e)
                    })
    
    def calculate_statistics(self):
        """Calculate aggregate statistics"""
        if self.metadata["image_stats"]["sizes"]:
            widths = [s[0] for s in self.metadata["image_stats"]["sizes"]]
            heights = [s[1] for s in self.metadata["image_stats"]["sizes"]]
            
            self.metadata["image_stats"]["mean_width"] = int(sum(widths) / len(widths))
            self.metadata["image_stats"]["mean_height"] = int(sum(heights) / len(heights))
            self.metadata["image_stats"]["min_width"] = min(widths)
            self.metadata["image_stats"]["max_width"] = max(widths)
            self.metadata["image_stats"]["min_height"] = min(heights)
            self.metadata["image_stats"]["max_height"] = max(heights)
        
        # Remove size list (too large for JSON)
        del self.metadata["image_stats"]["sizes"]
        
        # Convert defaultdict to dict
        self.metadata["image_stats"]["formats"] = dict(self.metadata["image_stats"]["formats"])
        
        # Calculate class balance
        if self.metadata["classes"]:
            class_counts = [c["count"] for c in self.metadata["classes"].values()]
            self.metadata["balance_ratio"] = min(class_counts) / max(class_counts) if class_counts else 0
    
    def generate(self):
        """Generate complete metadata"""
        print("🔍 Generating Dataset Metadata for PlantaeClassScanner")
        print("=" * 60)
        
        # Analyze each split
        self.analyze_directory(self.data_dir / "train", "train")
        self.analyze_directory(self.data_dir / "val", "val")
        self.analyze_directory(self.data_dir / "test", "test")
        self.analyze_directory(self.data_dir / "raw")
        
        # Calculate stats
        self.calculate_statistics()
        
        # Save metadata
        output_path = self.data_dir / "dataset_metadata.json"
        with open(output_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n✅ Metadata saved to: {output_path}")
        self.print_summary()
    
    def print_summary(self):
        """Print human-readable summary"""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total Images: {self.metadata['total_images']}")
        print(f"Number of Classes: {len(self.metadata['classes'])}")
        print(f"Total Size: {self.metadata['image_stats']['total_size_mb']:.2f} MB")
        print(f"Balance Ratio: {self.metadata.get('balance_ratio', 0):.2%}")
        
        print(f"\n📊 Class Distribution:")
        for class_name, stats in sorted(self.metadata["classes"].items()):
            print(f"  {class_name}: {stats['count']} images")
        
        print(f"\n🔀 Split Distribution:")
        for split, stats in self.metadata["splits"].items():
            print(f"  {split}: {stats['count']} images ({stats['percentage']}%)")
        
        print(f"\n⚠️  Quality Issues:")
        print(f"  Corrupt Images: {len(self.metadata['quality_checks']['corrupt_images'])}")
        print(f"  Duplicate Images: {len(self.metadata['quality_checks']['duplicate_images'])}")
        print(f"  Low Resolution: {len(self.metadata['quality_checks']['low_resolution'])}")

if __name__ == "__main__":
    generator = MetadataGenerator()
    generator.generate()