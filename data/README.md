## Overview
This directory contains the dataset structure and metadata for plant classification. Raw images are stored via Git LFS and can be downloaded using the provided scripts.

## Dataset Statistics
- **Version:** 1.0.0
- **Total Images:** TBD
- **Number of Classes:** 10 (initial - 10/2025)
- **Train/Val/Test Split:** 70/15/15
- **Image Format:** JPEG
- **Resolution Range:** 224x224 to 1024x1024

## Directory Structure
**data/**
   ├── raw/              # Original images (Git LFS tracked) \
   ├── processed/        # Preprocessed/augmented images \
   ├── train/            # Training set (70%) \
   ├── val/              # Validation set (15%) \
   ├── test/             # Test set (15%) \
   ├── user_collected/   # User-contributed images via scanner \
   └── dataset_metadata.json  # Detailed statistics \
   
## Data Sources
1. **iNaturalist API** - Research-grade observations
2. **PlantNet-300K** - Subset of high-quality images
3. **User Contributions** - In-field scans from device

## Class List (v1.0)
1. Monstera deliciosa (Swiss Cheese Plant)
2. Epipremnum aureum (Pothos)
3. Sansevieria trifasciata (Snake Plant)
4. Chlorophytum comosum (Spider Plant)
5. Spathiphyllum (Peace Lily)
6. Ficus lyrata (Fiddle Leaf Fig)
7. Aloe vera
8. Crassula ovata (Jade Plant)
9. Hedera helix (English Ivy)
10. Ficus elastica (Rubber Plant)

## Data Collection
Run the automated collection script:  
```Ruby
python scripts/download_dataset.py --species-list config/species.json --images-per-class 100
```

## Preprocessing
Images are preprocessed using:
Resize to 224x224 (MobileNetV3 input size) \
Normalization: [0, 1] range \
Augmentation: rotation (+15°/-15°), horizontal flip, brightness (+20% / -20%) \

## Dataset Metadata Schema
```Ruby
{
  "version": "1.0.0",
  "created_date": "2025-10-08",
  "total_images": 1500,
  "classes": {
    "Monstera_deliciosa": {
      "count": 150,
      "train": 105,
      "val": 22,
      "test": 23,
      "source": "inaturalist",
      "verified": true
    }
  },
  "image_stats": {
    "mean_width": 512,
    "mean_height": 512,
    "format": "JPEG"
  }
}
```

## License
Images sourced from iNaturalist are licensed under CC BY-NC 4.0. \
User-contributed images are owned by the contributor. \
